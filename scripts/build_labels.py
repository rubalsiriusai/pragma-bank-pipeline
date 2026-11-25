"""Build pragmatiq label tables from real outcomes in the raw bank tables.

Every label is a forecast task with an ``eval_ts`` column. pragmatiq cuts each
user's history at ``eval_ts`` before embedding, so the outcome window (everything
after ``eval_ts``) can never leak into the model input, even though the fraud and
collections tables also feed the event stream.

Tasks written to ``<out>/labels/``:

  default_12m   the customer has a collections case opened within 12 months
  fraud_12m     the customer has a fraud alert raised within 12 months
  churn_6m      the customer's relationship closes within 6 months

Population: customers whose relationship opened before ``eval_ts`` and had not
closed by it. Customers already in collections before ``eval_ts`` are left out of
``default_12m``, so the task predicts first entry into collections.

Usage:
    python scripts/build_labels.py --raw data/raw --out data/processed --eval-ts 2027-10-01
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

FORECAST_SCHEMA = pa.schema([
    pa.field("user_id", pa.string(), nullable=False),
    pa.field("eval_ts", pa.timestamp("us"), nullable=False),
    pa.field("label", pa.int8(), nullable=False),
])


def _scan(raw: Path, table: str) -> pl.LazyFrame:
    return pl.scan_parquet(str(raw / table / "*.parquet"))


def population(raw: Path, eval_ts: datetime) -> pl.DataFrame:
    """Customers open at ``eval_ts``: opened before it and not closed by it."""
    return (_scan(raw, "customer_profile")
            .filter(pl.col("relationship_open_date").cast(pl.Datetime("us")) < eval_ts)
            .filter(pl.col("relationship_close_date").is_null()
                    | (pl.col("relationship_close_date").cast(pl.Datetime("us")) >= eval_ts))
            .select(pl.col("customer_id").alias("user_id"))
            .unique().sort("user_id").collect())


def _outcome(pop: pl.DataFrame, hits: pl.DataFrame, eval_ts: datetime) -> pl.DataFrame:
    """Label 1 for population members in ``hits``, 0 for the rest."""
    hit_ids = hits.select("user_id").unique().with_columns(pl.lit(1, pl.Int8).alias("label"))
    return (pop.join(hit_ids, on="user_id", how="left")
            .with_columns(pl.col("label").fill_null(0).cast(pl.Int8),
                          pl.lit(eval_ts).cast(pl.Datetime("us")).alias("eval_ts"))
            .select("user_id", "eval_ts", "label"))


def build(raw: Path, eval_ts: datetime) -> dict[str, pl.DataFrame]:
    """All label tables, keyed by task name."""
    pop = population(raw, eval_ts)
    in_12m = eval_ts + timedelta(days=365)
    in_6m = eval_ts + timedelta(days=182)

    coll = _scan(raw, "collections_case").select(
        pl.col("customer_id").alias("user_id"), pl.col("case_open_timestamp").alias("t")).collect()
    already = coll.filter(pl.col("t") <= eval_ts).select("user_id").unique()
    default_pop = pop.join(already, on="user_id", how="anti")
    default = _outcome(default_pop, coll.filter((pl.col("t") > eval_ts) & (pl.col("t") <= in_12m)),
                       eval_ts)

    fraud = _scan(raw, "fraud_case").select(
        pl.col("customer_id").alias("user_id"), pl.col("alert_timestamp").alias("t")).collect()
    fraud_lab = _outcome(pop, fraud.filter((pl.col("t") > eval_ts) & (pl.col("t") <= in_12m)),
                         eval_ts)

    closes = _scan(raw, "customer_profile").select(
        pl.col("customer_id").alias("user_id"),
        pl.col("relationship_close_date").cast(pl.Datetime("us")).alias("t")).collect()
    churn = _outcome(pop, closes.filter((pl.col("t") > eval_ts) & (pl.col("t") <= in_6m)), eval_ts)

    return {"default_12m": default, "fraud_12m": fraud_lab, "churn_6m": churn}


def write(tables: dict[str, pl.DataFrame], out: Path) -> dict[str, dict[str, float]]:
    """Write each table to ``out/labels/<task>.parquet``; returns per-task stats."""
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for task, df in tables.items():
        table = df.to_arrow().cast(FORECAST_SCHEMA)
        pq.write_table(table, labels_dir / f"{task}.parquet")
        n, pos = df.height, int(df["label"].sum())
        stats[task] = {"rows": n, "positives": pos, "rate": pos / n if n else 0.0}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--eval-ts", type=str, default="2027-10-01")
    args = ap.parse_args()
    stats = write(build(args.raw, datetime.fromisoformat(args.eval_ts)), args.out)
    for task, s in stats.items():
        print(f"  {task:14s} {s['rows']:6d} customers  {s['positives']:5d} positive  "
              f"({s['rate']:.1%})")


if __name__ == "__main__":
    main()
