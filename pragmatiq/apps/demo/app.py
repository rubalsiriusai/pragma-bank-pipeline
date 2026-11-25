"""pragmatiq demo.

Pick a real customer → see their event timeline → the model's user embedding
(attach a fine-tuned head via ``pragmatiq finetune`` for calibrated
fraud / credit / churn scores).

Run:  streamlit run apps/demo/app.py
Env:  PRAGMATIQ_OUT   (pipeline output dir, default runs/pipeline) — the demo reads the
                      trained run from <out>/runs/pragmatiq.
      PRAGMATIQ_RAW   (contract data dir, default data/processed — the intake output)
      PRAGMATIQ_RUN   (override the run path on its own)

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

# `pragmatiq pipeline <data> --out <OUT>` writes the trained run to <OUT>/runs/pragmatiq;
# the events come from the contract directory the pipeline read (the intake output).
OUT = Path(os.environ.get("PRAGMATIQ_OUT", "runs/pipeline"))
_pipeline_run = OUT / "runs" / "pragmatiq"
RUN = Path(os.environ.get("PRAGMATIQ_RUN") or (_pipeline_run if _pipeline_run.exists() else OUT))
RAW = Path(os.environ.get("PRAGMATIQ_RAW", "data/processed"))


@st.cache_resource
def load_model():
    from pragmatiq.models.pragmatiq import PragmaModel

    return PragmaModel.from_pretrained(RUN)


@st.cache_data
def load_users() -> list[str]:
    ids = pq.read_table(RAW / "profiles.parquet", columns=["user_id"]).column("user_id")
    return sorted(ids.to_pylist())


@st.cache_data
def load_user_events(user_id: str) -> pd.DataFrame:
    # Real datasets hold tens of millions of events: read one customer, not the file.
    df = pq.read_table(RAW / "events.parquet", filters=[("user_id", "==", user_id)]).to_pandas()
    df["fields"] = df["fields"].apply(dict)
    return df


def main() -> None:
    st.set_page_config(page_title="pragmatiq demo", layout="wide")
    st.title("pragmatiq — behavioral foundation model demo")
    st.caption("Independent implementation inspired by the PRAGMA paper (arXiv 2604.08649); "
               "not affiliated with or endorsed by Revolut.")

    if not RUN.exists() or not RAW.exists():
        st.warning(f"Point PRAGMATIQ_RUN ({RUN}) and PRAGMATIQ_RAW ({RAW}) at a trained run "
                   "and a contract dataset. Run `pragmatiq pipeline <data_dir>` first.")
        return

    users = load_users()[:500]
    user_id = st.sidebar.selectbox("Customer", users)
    ev = load_user_events(user_id).sort_values("ts")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Event timeline — {user_id} ({len(ev)} events)")
        show = ev.copy()
        show["fields"] = show["fields"].apply(lambda f: ", ".join(f"{k}={v}" for k, v in f.items()))
        st.dataframe(show[["ts", "source", "fields"]].tail(50), use_container_width=True)

    with col2:
        st.subheader("Model scores")
        try:
            model = load_model()
            recs = [{
                "user_id": user_id,
                "events": [(int(pd.Timestamp(r.ts).value // 1000), r.source, dict(r.fields))
                           for r in ev.itertuples()],
                "attributes": {}, "lifelong": [],
            }]
            emb = model.embed_records(recs)
            st.metric("embedding norm", f"{np.linalg.norm(emb[0]):.2f}")
            st.caption("Attach fine-tuned heads (fraud/credit/churn) for calibrated scores; "
                       "this demo shows the raw embedding.")
        except Exception as exc:  # pragma: no cover - demo resilience
            st.error(f"model unavailable: {exc}")

    transfers_path = RAW / "transfers.parquet"
    if transfers_path.exists():
        st.subheader("Ego transfer graph")
        tr = pq.read_table(transfers_path).to_pandas()
        ego = tr[(tr["from_user"] == user_id) | (tr["to_user"] == user_id)]
        st.write(f"{len(ego)} transfers involving {user_id}")
        st.dataframe(ego[["from_user", "to_user", "amount", "ts"]].head(30), use_container_width=True)


if __name__ == "__main__":
    main()
