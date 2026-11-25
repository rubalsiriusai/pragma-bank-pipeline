"""Run the whole flow on the real bank data, start to finish.

  1. intake check     score data/raw against the mapping (gate + readiness)
  2. intake run       data/raw  ->  data/processed/{events,profiles}.parquet
  3. labels           real outcomes -> data/processed/labels/*.parquet
  4. pragmatiq        validate -> tokenize -> pretrain -> embed -> probe
                      (outputs/pragmatiq/: tok/, runs/, embeddings.parquet,
                       pipeline_summary.json)

Each stage can be skipped when its output already exists (``--skip-intake``,
``--skip-labels``), so the model can be retrained without re-running intake.

Usage (from the project root):
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-intake --max-steps 800 --model-size small
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(*args: str) -> None:
    """Run a command from the project root, streaming its output; stop on failure."""
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--mapping", default="configs/intake_mapping.yaml")
    ap.add_argument("--now", default="2028-10-13",
                    help="intake clock; must be after the data's last event")
    ap.add_argument("--eval-ts", default="2027-10-01", help="label eval point")
    ap.add_argument("--out", default="outputs/pragmatiq")
    ap.add_argument("--tokenizer-config", default="configs/pragmatiq_tokenizer.yaml")
    ap.add_argument("--model-size", default="nano")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-workers", type=int, default=4, help="parallel tokenize workers")
    ap.add_argument("--skip-intake", action="store_true")
    ap.add_argument("--skip-labels", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    t0 = time.time()
    if not args.skip_intake:
        sh(py, "-m", "intake.cli", "--device", args.device, "check", args.raw,
           "--mapping", args.mapping, "--now", args.now, "--out", "outputs/intake_check")
        sh(py, "-m", "intake.cli", "--device", args.device, "run", args.raw,
           "--mapping", args.mapping, "--out", args.processed, "--now", args.now,
           "--work-dir", "outputs/intake_work")
    if not args.skip_labels:
        sh(py, "scripts/build_labels.py", "--raw", args.raw, "--out", args.processed,
           "--eval-ts", args.eval_ts)

    sh(py, "-m", "pragmatiq.cli", "validate", args.processed)
    sh(py, "-m", "pragmatiq.cli", "pipeline", args.processed, "--out", args.out,
       "--model-size", args.model_size, "--max-steps", str(args.max_steps),
       "--token-budget", str(args.token_budget), "--tokenizer-config", args.tokenizer_config,
       "--device", args.device, "--n-workers", str(args.n_workers))

    summary = json.loads((ROOT / args.out / "pipeline_summary.json").read_text(encoding="utf-8"))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    for task, res in summary["probes"].items():
        print(f"  {task:14s} probe AUC {res['probe_auc']:.3f}   "
              f"baseline AUC {res.get('baseline_auc', float('nan')):.3f}")


if __name__ == "__main__":
    main()
