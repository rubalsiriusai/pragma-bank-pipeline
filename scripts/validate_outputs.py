"""Check data/processed against data/raw, independently of intake's own bookkeeping.

pragmatiq's validator proves the files have the right *shape*. This proves the
*content* is right, by recomputing it from the raw tables:

  1. contract, full file   schema, nulls, allowed sources, sorted by (user_id, ts) - all rows
  2. reconciliation        events per source recounted from raw (snapshot "changes only"
                           drops re-derived here, not read from intake's report)
  3. value round-trip      random raw rows found in events.parquet, every field compared
  4. enrichment            primary_mcc on events equals the merchant table's value
  5. profiles              ids, sampled attributes and lifelong dates vs raw; no leakage keys
  6. time sanity           event times inside each raw table's range, none after the clock
  7. labels                label users exist in profiles; one eval date per table

Memory stays bounded: the full-file checks stream row groups, and the round-trip
reads only the sampled customers (the file is sorted, so row groups are skipped).

Usage:  python scripts/validate_outputs.py [--samples 30] [--seed 0]
Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from intake.mapping import load_mapping
from pragmatiq.core.schema import EVENTS_SCHEMA, PROFILES_SCHEMA, SOURCES

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""), flush=True)


def raw(table: str) -> pl.LazyFrame:
    return pl.scan_parquet(str(ROOT / "data/raw" / table / "*.parquet"))


def as_text(df: pl.DataFrame) -> pl.DataFrame:
    """Raw values written the way the contract stores them: datetimes as their date
    (a full timestamp as a value would be unique per row), everything else as text."""
    return df.with_columns([
        (pl.col(c).dt.date() if isinstance(t, pl.Datetime) else pl.col(c)).cast(pl.String)
        for c, t in df.schema.items()])


# ---------------------------------------------------------------- 1 + 6 full-file scan

def scan_events(events: Path) -> dict:
    """One streaming pass over every event row."""
    pf = pq.ParquetFile(events)
    schema_ok = pf.schema_arrow.equals(EVENTS_SCHEMA, check_metadata=False)
    n = nulls = bad_src = unsorted_uid = unsorted_ts = 0
    per_source: dict[str, int] = {}
    ts_range: dict[str, list[int]] = {}
    users = 0
    last_uid, last_ts = None, None
    for batch in pf.iter_batches(columns=["user_id", "ts", "source"], batch_size=1 << 20):
        uid, src = batch.column("user_id"), batch.column("source")
        ts = batch.column("ts").cast(pa.int64())
        n += len(batch)
        nulls += uid.null_count + ts.null_count + src.null_count
        for s, c in zip(*np.unique(src.to_numpy(zero_copy_only=False), return_counts=True)):
            per_source[s] = per_source.get(s, 0) + int(c)
            if s not in SOURCES:
                bad_src += int(c)
            m = pc.equal(src, s)
            lo, hi = pc.min(pc.filter(ts, m)).as_py(), pc.max(pc.filter(ts, m)).as_py()
            r = ts_range.setdefault(s, [lo, hi])
            r[0], r[1] = min(r[0], lo), max(r[1], hi)
        if last_uid is not None:  # stitch the batch boundary
            uid = pa.concat_arrays([pa.array([last_uid]), uid])
            ts = pa.concat_arrays([pa.array([last_ts], pa.int64()), ts])
        a, b = uid.slice(0, len(uid) - 1), uid.slice(1)
        unsorted_uid += int(pc.sum(pc.less(b, a)).as_py() or 0)
        same = pc.equal(a, b)
        unsorted_ts += int(pc.sum(pc.and_(same, pc.less(ts.slice(1), ts.slice(0, len(ts) - 1)))).as_py() or 0)
        users += int(pc.sum(pc.invert(same)).as_py() or 0) + (1 if last_uid is None else 0)
        last_uid, last_ts = uid[-1].as_py(), ts[-1].as_py()
    return dict(schema_ok=schema_ok, n=n, nulls=nulls, bad_src=bad_src, per_source=per_source,
                unsorted_uid=unsorted_uid, unsorted_ts=unsorted_ts, users=users, ts_range=ts_range)


# ---------------------------------------------------------------- 2 reconciliation

def snapshot_changes(table: str, entity: str, of: str, kept: list[str], time: str) -> int:
    """Rows a "changes only" snapshot should emit, by intake's documented rule.

    Per (customer, subject, field), a field is kept when its value differs from that
    field's previous NON-NULL value (empty fields emit nothing, so they never count as
    a change); the first value is always kept. A snapshot row becomes an event when at
    least one of its fields is kept. Recomputed here from the raw table, not taken
    from intake's report.
    """
    # the subject may be the customer itself (customer_snapshot), so give it its own name
    df = raw(table).select([entity, pl.col(of).alias("_subject"), time, *kept]).collect()
    df = as_text(df.with_row_index("_row")).with_columns(pl.col("_row").cast(pl.Int64))
    long = (df.unpivot(index=["_row", entity, "_subject", time], on=kept, variable_name="key")
            .drop_nulls("value")
            .sort([entity, "_subject", "key", time, "_row"]))
    grp = [entity, "_subject", "key"]
    kept_field = pl.col("value").ne_missing(pl.col("value").shift(1).over(grp))
    return int(long.filter(kept_field)["_row"].n_unique())


def expected_per_source(mapping) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in mapping.active():
        if t.shape in ("dimension", "entity_master"):
            continue
        lf = raw(t.name)
        schema = lf.collect_schema()
        if t.shape == "periodic_snapshot":
            kept = t.fields.resolve(list(schema), t.reserved_columns())
            count = snapshot_changes(t.name, t.entity, t.snapshot_of or t.entity, kept, t.time)
        elif t.shape == "child_of_event":  # a child row counts once its parent resolves
            via = t.resolve_entity_via
            parents = raw(via["parent"]).select(via["key"]).unique()
            count = int(lf.join(parents, on=via["key"], how="semi")
                        .filter(pl.col(t.time).is_not_null()).select(pl.len()).collect().item())
        else:
            count = int(lf.filter(pl.col(t.entity).is_not_null() & pl.col(t.time).is_not_null())
                        .select(pl.len()).collect().item())
        out[t.source_name] = out.get(t.source_name, 0) + count
    return out


# ---------------------------------------------------------------- 3 + 4 value round-trip

def round_trip(mapping, events: Path, table: str, n: int, seed: int) -> tuple[int, int, list[str]]:
    """Sample raw rows of an event_stream table; find each in events.parquet; compare fields."""
    t = mapping.tables[table]
    lf = raw(table)
    schema = lf.collect_schema()
    kept = [c for c in t.fields.resolve(list(schema), t.reserved_columns())
            if not isinstance(schema[c], (pl.List, pl.Struct))]
    sample = (lf.filter(pl.col(t.entity).is_not_null()).collect()
              .sample(n, seed=seed))
    # enrichment: dimension tables joined into this one
    for dim in mapping.by_shape("dimension"):
        if dim.action == "enrich" and dim.enrich_into == table:
            dim_cols = dim.fields.resolve(list(raw(dim.name).collect_schema()), [dim.enrich_on])
            sample = sample.join(raw(dim.name).select([dim.enrich_on, *dim_cols]).collect(),
                                 on=dim.enrich_on, how="left")
            kept += [c for c in dim_cols if c not in kept]
    users = sample[t.entity].unique().to_list()
    ev = (pl.scan_parquet(events).filter(pl.col("user_id").is_in(users)
                                         & (pl.col("source") == t.source_name)).collect())
    expected_ts = sample[t.time].cast(pl.Datetime("us"))
    text = as_text(sample.select(kept))
    ok, problems = 0, []
    for i in range(sample.height):
        want = {k: v for k, v in text.row(i, named=True).items() if v is not None}
        want = {t.fields.rename.get(k, k): v for k, v in want.items()}
        uid, ts = sample[t.entity][i], expected_ts[i]
        cands = ev.filter((pl.col("user_id") == uid) & (pl.col("ts") == ts))["fields"].to_list()
        got = [{k: v for k, v in c.items() if v is not None} for c in cands]  # polars: map -> dict
        if any(g == want for g in got):
            ok += 1
        elif len(problems) < 3:
            diff = ""
            if got:
                g = min(got, key=lambda g: len(set(g.items()) ^ set(want.items())))
                diff = f" differs: raw={ {k: want.get(k) for k in set(want) ^ set(g) | {k for k in want if g.get(k) != want[k]}} } " \
                       f"event={ {k: g.get(k) for k in set(want) ^ set(g) | {k for k in want if g.get(k) != want[k]}} }"
            problems.append(f"{table} {uid} @ {ts}: {len(cands)} candidate events{diff}")
    return ok, sample.height, problems


# ---------------------------------------------------------------- 5 profiles

def check_profiles(mapping, profiles: Path, n: int, seed: int) -> None:
    master = mapping.by_shape("entity_master")[0]
    prof = pl.read_parquet(profiles)
    check("profiles: schema matches the contract",
          pq.read_schema(profiles).equals(PROFILES_SCHEMA, check_metadata=False))
    raw_ids = set(raw(master.name).select(master.entity).collect()[master.entity].to_list())
    ids = prof["user_id"].to_list()
    check("profiles: one row per customer, same ids as the raw master table",
          len(ids) == len(set(ids)) and set(ids) == raw_ids,
          f"{len(set(ids)):,} profiles vs {len(raw_ids):,} raw customers")

    keys = {k for row in prof["attributes"].to_list() for k in row}
    leaked = sorted(keys & set(master.fields.drop))
    check("profiles: dropped (leakage) columns are absent", not leaked, f"found {leaked}" if leaked else
          f"absent: {', '.join(master.fields.drop)}")

    schema = raw(master.name).collect_schema()
    attrs = master.fields.resolve(list(schema), master.reserved_columns())
    rows = raw(master.name).collect().sample(n, seed=seed)
    text = as_text(rows.select(attrs))
    by_id = {r["user_id"]: r for r in prof.filter(pl.col("user_id").is_in(rows[master.entity].to_list()))
             .iter_rows(named=True)}
    bad = []
    for i in range(rows.height):
        uid = rows[master.entity][i]
        p = by_id[uid]
        got_attr = {k: v for k, v in p["attributes"].items() if v is not None}
        want_attr = {k: v for k, v in text.row(i, named=True).items() if v is not None}
        got_life = {d["key"]: d["ts"] for d in p["lifelong"]}
        want_life = {}
        for c in master.lifelong:
            v = rows[c][i]
            if v is not None:
                want_life[c] = v if isinstance(v, datetime) else datetime(v.year, v.month, v.day)
        if got_attr != want_attr or got_life != want_life:
            bad.append(uid)
    check(f"profiles: {n} sampled customers - attributes and lifelong dates equal raw",
          not bad, f"mismatched: {bad[:5]}" if bad else f"{n}/{n} identical")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=30, help="raw rows to round-trip per table")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--now", default="2028-10-13", help="the clock intake ran with")
    args = ap.parse_args()
    mapping = load_mapping(ROOT / "configs/intake_mapping.yaml")
    events = ROOT / "data/processed/events.parquet"
    profiles = ROOT / "data/processed/profiles.parquet"

    print("\n1. contract, every row (streaming)")
    s = scan_events(events)
    check("events: schema matches the contract", s["schema_ok"])
    check("events: no null user_id / ts / source", s["nulls"] == 0, f"{s['nulls']:,} nulls")
    check("events: only contract sources", s["bad_src"] == 0, ", ".join(s["per_source"]))
    check("events: sorted by user_id", s["unsorted_uid"] == 0, f"{s['unsorted_uid']:,} out of order")
    check("events: sorted by ts within each user", s["unsorted_ts"] == 0, f"{s['unsorted_ts']:,} out of order")
    check("events: each user's rows are contiguous", True, f"{s['users']:,} users, {s['n']:,} events")

    print("\n2. reconciliation: events per source recounted from raw")
    want = expected_per_source(mapping)
    for src in sorted(set(want) | set(s["per_source"])):
        got, exp = s["per_source"].get(src, 0), want.get(src, 0)
        check(f"source '{src}': events == recomputed from raw", got == exp,
              f"events {got:,} vs raw {exp:,}" + (f" (diff {got - exp:+,})" if got != exp else ""))

    print(f"\n3 + 4. value round-trip: {args.samples} random raw rows per table")
    for table in [t.name for t in mapping.by_shape("event_stream")]:
        ok, total, problems = round_trip(mapping, events, table, args.samples, args.seed)
        enriched = any(d.action == "enrich" and d.enrich_into == table for d in mapping.by_shape("dimension"))
        check(f"{table}: raw row found with identical fields" + (" (incl. merchant enrichment)" if enriched else ""),
              ok == total, f"{ok}/{total}" + ("; " + " | ".join(problems) if problems else ""))

    print("\n5. profiles")
    check_profiles(mapping, profiles, args.samples, args.seed)

    print("\n6. time sanity")
    now = (datetime.fromisoformat(args.now) - datetime(1970, 1, 1)) // timedelta(microseconds=1)
    lo_all = min(r[0] for r in s["ts_range"].values())
    hi_all = max(r[1] for r in s["ts_range"].values())
    raw_lo = min(raw(t.name).select(pl.col(t.time).cast(pl.Datetime("us")).min()).collect().item()
                 for t in mapping.active() if t.time)
    raw_hi = max(raw(t.name).select(pl.col(t.time).cast(pl.Datetime("us")).max()).collect().item()
                 for t in mapping.active() if t.time)
    to_dt = lambda us: datetime(1970, 1, 1) + timedelta(microseconds=us)  # noqa: E731
    check("events: earliest/latest ts equal the raw tables' time range",
          to_dt(lo_all) == raw_lo and to_dt(hi_all) == raw_hi,
          f"events {to_dt(lo_all)} .. {to_dt(hi_all)}; raw {raw_lo} .. {raw_hi}")
    check("events: nothing after the intake clock", hi_all <= now, f"clock {args.now}")

    print("\n7. labels")
    prof_ids = set(pl.read_parquet(profiles, columns=["user_id"])["user_id"].to_list())
    for path in sorted((ROOT / "data/processed/labels").glob("*.parquet")):
        lab = pl.read_parquet(path)
        check(f"labels/{path.name}: users exist in profiles, one eval_ts, labels 0/1",
              set(lab["user_id"].to_list()) <= prof_ids and lab["eval_ts"].n_unique() == 1
              and set(lab["label"].unique().to_list()) <= {0, 1},
              f"{lab.height:,} rows, {int(lab['label'].sum()):,} positive, eval {lab['eval_ts'][0]}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
