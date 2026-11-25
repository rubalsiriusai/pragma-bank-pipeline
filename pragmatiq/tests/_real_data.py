"""Real-data test fixture: small datasets cut from real bank customers.

``tests/fixtures/real_sample/`` holds a few hundred real customers that went through
the intake package (``events.parquet`` + ``profiles.parquet``) and their real outcome
labels (``labels/{default_12m,fraud_12m,churn_6m}.parquet``). Each customer's event
history is thinned to every k-th event so the fixture stays small while still
spanning the full time range. It is rebuilt with ``scripts/build_test_fixture.py``
in the project root.

Tests ask for a dataset with :func:`real_sample`, which writes a deterministic
subset of those customers (chosen by ``seed``) into a contract directory, so the
code under test always runs on real records, never generated ones.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "real_sample"

# Label task the old generator-based tests used as their "credit" task.
CREDIT_TASK = "default_12m"


@dataclass
class SampleConfig:
    """How many real customers to cut, and which (``seed``). Other keys are accepted
    and ignored, so call sites can keep descriptive settings next to the size."""

    n_users: int = 100
    seed: int = 0

    def __init__(self, n_users: int = 100, seed: int = 0, **_ignored: Any) -> None:
        self.n_users = n_users
        self.seed = seed


def fixture_users() -> list[str]:
    """Every customer id in the fixture, sorted."""
    ids = pq.read_table(FIXTURE_DIR / "profiles.parquet", columns=["user_id"]).column("user_id")
    return sorted(ids.to_pylist())


def real_sample(cfg: SampleConfig | dict[str, Any] | None = None, out: str | Path | None = None,
                **_ignored: Any) -> dict[str, Any]:
    """Write ``cfg.n_users`` real customers (chosen by ``cfg.seed``) to ``out``.

    ``cfg`` is a :class:`SampleConfig` or a dict with ``n_users`` / ``seed``.
    Writes ``events.parquet``, ``profiles.parquet``, ``labels/*.parquet`` and a
    ``manifest.json``; returns the manifest. Events keep the (user_id, ts) order
    of the fixture. Asking for more customers than the fixture has returns all.
    """
    import json

    if cfg is None:
        cfg = SampleConfig()
    elif isinstance(cfg, dict):
        cfg = SampleConfig(**cfg)
    if out is None:
        raise TypeError("real_sample() needs an output directory")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    users = fixture_users()
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.n_users, len(users))
    chosen = sorted(rng.choice(users, size=n, replace=False).tolist())
    keep = pa.array(chosen, type=pa.string())

    def _filter(src: Path, dst: Path, col: str = "user_id") -> int:
        t = pq.read_table(src)
        t = t.filter(pc.is_in(t[col], value_set=keep))
        pq.write_table(t, dst)
        return t.num_rows

    n_events = _filter(FIXTURE_DIR / "events.parquet", out / "events.parquet")
    _filter(FIXTURE_DIR / "profiles.parquet", out / "profiles.parquet")
    labels_out = out / "labels"
    labels_out.mkdir(exist_ok=True)
    for lab in sorted((FIXTURE_DIR / "labels").glob("*.parquet")):
        _filter(lab, labels_out / lab.name)
    manifest = {"source": "real_sample", "n_users": n, "n_events": n_events, "seed": cfg.seed}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


MEMO_KEY = "memo"


def with_memo_text(data_dir: str | Path) -> Path:
    """Give every transaction event a free-text ``memo`` built from its own real fields.

    The bank tables carry no free text (every string column is a code), so the
    tokenizer's text paths (BPE pieces, the text-embed variant, null text values)
    would go unexercised. The memo is a statement-style descriptor assembled
    from the event's real ``transaction_type``, ``category``, ``primary_mcc`` and
    ``merchant_country`` values; it adds no information that is not already there.
    Rewrites ``events.parquet`` in place and returns ``data_dir``.
    """
    from pragmatiq.core.schema import EVENTS_SCHEMA

    path = Path(data_dir) / "events.parquet"
    t = pq.read_table(path)
    fields = t.column("fields").to_pylist()
    out = []
    for src, kv in zip(t.column("source").to_pylist(), fields):
        d = dict(kv)
        if src == "transaction" and "transaction_type" in d:
            parts = [d.get("transaction_type"), d.get("category"),
                     f"MCC {d['primary_mcc']}" if d.get("primary_mcc") else None,
                     d.get("merchant_country")]
            d[MEMO_KEY] = " ".join(p for p in parts if p)
        out.append(list(d.items()))
    t = t.set_column(t.schema.get_field_index("fields"), "fields",
                     pa.array(out, type=EVENTS_SCHEMA.field("fields").type))
    pq.write_table(t.cast(EVENTS_SCHEMA), path)
    return Path(data_dir)


def copy_fixture(out: str | Path) -> Path:
    """Copy the whole fixture to ``out`` (for tests that mutate files in place)."""
    shutil.copytree(FIXTURE_DIR, out, dirs_exist_ok=True)
    return Path(out)
