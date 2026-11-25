"""Cut the small real-data fixture pragmatiq's test suite runs on.

Reads the intake output (``data/processed``: events, profiles, labels) and writes
``pragmatiq/tests/fixtures/real_sample/``:

* ``--users`` real customers, sampled so every label task keeps some positives
  (positives are rare on the full book, ~1-6%, which would leave a small test
  sample with none);
* each customer's events thinned to at most ``--max-events`` by keeping every
  k-th event, so the history still spans the whole time range (and both sides
  of the label eval point) while the fixture stays a few MB;
* their profiles and their rows of every label table.

Usage:
    python scripts/build_test_fixture.py --data data/processed \\
        --out pragmatiq/tests/fixtures/real_sample --users 400 --max-events 160
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from pragmatiq.core.schema import EVENTS_SCHEMA, PROFILES_SCHEMA


def choose_users(labels_dir: Path, n_users: int, per_task_pos: int, seed: int) -> list[str]:
    """Positives from every task first (up to ``per_task_pos`` each), then fill with others."""
    chosen: set[str] = set()
    pool: set[str] = set()
    for path in sorted(labels_dir.glob("*.parquet")):
        df = pl.read_parquet(path)
        pos = df.filter(pl.col("label") == 1)["user_id"]
        chosen |= set(pos.sample(min(per_task_pos, len(pos)), seed=seed).to_list())
        pool |= set(df["user_id"].to_list())
    rest = sorted(pool - chosen)
    need = max(n_users - len(chosen), 0)
    fill = pl.Series(rest).sample(min(need, len(rest)), seed=seed).to_list()
    return sorted(chosen | set(fill))[:n_users]


def _write(df: pl.DataFrame, schema: pa.Schema, path: Path) -> None:
    pq.write_table(df.to_arrow().select(schema.names).cast(schema), path, compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path,
                    default=Path("pragmatiq/tests/fixtures/real_sample"))
    ap.add_argument("--users", type=int, default=400)
    ap.add_argument("--max-events", type=int, default=160)
    ap.add_argument("--per-task-positives", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    users = choose_users(args.data / "labels", args.users, args.per_task_positives, args.seed)
    keep = users  # a plain list: is_in(Series) of the same dtype is deprecated

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "labels").mkdir(parents=True)

    events = (pl.scan_parquet(args.data / "events.parquet")
              .filter(pl.col("user_id").is_in(keep))
              .with_columns(pl.int_range(pl.len()).over("user_id").alias("_i"),
                            pl.len().over("user_id").alias("_n"))
              # every k-th event, k chosen per user so at most max_events remain
              .filter(pl.col("_i") % ((pl.col("_n") + args.max_events - 1) // args.max_events) == 0)
              .drop("_i", "_n")
              .collect())
    # same order as the intake output: user_id, then time
    events = events.sort(["user_id", "ts"], maintain_order=True)
    # polars writes large_string; cast to the exact contract schema pragmatiq validates
    _write(events, EVENTS_SCHEMA, args.out / "events.parquet")

    profiles = (pl.read_parquet(args.data / "profiles.parquet")
                .filter(pl.col("user_id").is_in(keep)).sort("user_id"))
    _write(profiles, PROFILES_SCHEMA, args.out / "profiles.parquet")

    for path in sorted((args.data / "labels").glob("*.parquet")):
        df = pl.read_parquet(path).filter(pl.col("user_id").is_in(keep)).sort("user_id")
        _write(df, pq.read_schema(path), args.out / "labels" / path.name)
        print(f"  {path.stem:14s} {df.height:4d} customers  {int(df['label'].sum()):3d} positive")

    size = sum(p.stat().st_size for p in args.out.rglob("*.parquet")) / 1e6
    print(f"  {len(users)} customers, {events.height:,} events, {size:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
