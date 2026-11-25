# pragma-bank-pipeline

Real bank tables go through the **intake** package, which turns them into the
pragmatiq data contract (`events.parquet` + `profiles.parquet`). That contract,
plus labels built from real outcomes, then flows through **pragmatiq**, an
implementation of the PRAGMA paper (arXiv 2604.08649) with its synthetic data
generator removed.

> pragmatiq is an independent implementation inspired by the PRAGMA paper
> (arXiv 2604.08649) and is not affiliated with or endorsed by Revolut.

## Layout

```
pragma-bank-pipeline/
├── intake/                      # client tables -> events/profiles contract (CLI: fluxtra-intake)
├── pragmatiq/                   # PRAGMA model: tokenizer, pretraining, embeddings, probes (CLI: pragmatiq)
├── configs/
│   ├── intake_mapping.yaml      # reviewed mapping: how each raw table becomes events/profiles
│   ├── intake_mapping.draft.yaml# what `intake discover` proposed, kept for comparison
│   └── pragmatiq_tokenizer.yaml # tokenizer settings for this data
├── scripts/
│   ├── vm_setup.sh              # GPU VM: environment with CUDA torch
│   ├── vm_get_data.sh           # GPU VM: raw tables from S3 -> data/raw
│   ├── vm_train.sh              # GPU VM: intake -> labels -> verify -> tokenize -> pretrain -> embed -> probe
│   ├── run_pipeline.py          # local end-to-end (small CPU run)
│   ├── build_labels.py          # real-outcome label tables (default / fraud / churn)
│   ├── validate_outputs.py      # checks events/profiles against the raw tables (24 checks)
│   ├── build_test_fixture.py    # cuts pragmatiq's real-data test fixture
│   └── make_report.py           # the PDF report
├── reports/                     # the PDF report
├── data/          (not in git)  # raw/ = 12 source tables, processed/ = intake output + labels/
└── outputs/       (not in git)  # intake reports, tokenized shards, model runs, embeddings, logs
```

Data never goes to GitHub: it is large (files over 100 MB) and customer-level bank
data. `.gitignore` excludes `data/`, `outputs/`, every `.parquet`, the virtual
environment and the real-customer test fixture.

Secrets never go in the code either: AWS keys live only in `aws configure` on the
machine that downloads the data, and the S3 location is passed at run time
(`S3_PATH=...`). `.gitignore` also blocks `.env` files, `*.pem` keys and `.aws/`.

## Run on a GPU VM (full training)

Prerequisites on the VM: NVIDIA driver (`nvidia-smi` works), `git`, and AWS CLI v2.

```bash
# 1. code (public repo: no login needed)
git clone https://github.com/rubalsiriusai/pragma-bank-pipeline.git
cd pragma-bank-pipeline

# 2. environment: .venv with CUDA torch + both packages; prints "device": "cuda"
bash scripts/vm_setup.sh

# 3. data: the 12 raw tables from S3 into data/raw/
aws configure                    # your access key, secret key, region: saved on the VM only
S3_PATH="s3://<your-bucket>/<your-prefix>/" bash scripts/vm_get_data.sh

# 4. everything else, in the background (survives closing the SSH session)
mkdir -p outputs
nohup bash scripts/vm_train.sh > outputs/vm_train.log 2>&1 &
tail -f outputs/vm_train.log     # Ctrl+C stops watching, not the run
```

`vm_train.sh` runs, in order:

| Step | What it does | Output |
|---|---|---|
| 1 intake | raw tables -> events + profiles (`--now 2028-10-13`) | `data/processed/events.parquet`, `profiles.parquet` |
| 2 labels | default / fraud / churn from real outcomes | `data/processed/labels/` |
| 3 verify | 24 checks against the raw tables; **stops if any fail** | printed PASS/FAIL |
| 4 tokenize | vocabulary + token shards | `outputs/tok/` |
| 5 pretrain | shows the plan, then trains; 1 epoch sized to the GPU by default | `outputs/runs/bank_small/` |
| 6 embed | one vector per customer | `outputs/embeddings_bank_small.parquet` |
| 7 probe | ROC-AUC per label vs a raw-count baseline | `outputs/probe_bank_small_*.json` |

Settings via environment variables, for example:

```bash
MODEL_SIZE=medium nohup bash scripts/vm_train.sh > outputs/vm_train.log 2>&1 &
PRETRAIN_CONFIG=pragmatiq/configs/pretrain.yaml bash scripts/vm_train.sh   # fixed steps instead of auto
```

`MODEL_SIZE` is `small` (~10M parameters, default), `medium` (~100M) or `large` (~1B).
Re-running is safe: finished stages are skipped (delete their output to redo one),
and pretraining resumes from its last checkpoint. Training progress is in
`outputs/runs/<run>/metrics.jsonl` (loss should fall); checkpoints in `checkpoints/`.

## Run locally (Windows, CPU)

```bash
# one-time: environment (CPU torch)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.6" -e "pragmatiq[train,dev]" -e intake reportlab

# the stages one at a time
.venv/Scripts/fluxtra-intake check data/raw --mapping configs/intake_mapping.yaml --now 2028-10-13
.venv/Scripts/fluxtra-intake run   data/raw --mapping configs/intake_mapping.yaml --out data/processed --now 2028-10-13
.venv/Scripts/python scripts/build_labels.py --raw data/raw --out data/processed --eval-ts 2027-10-01
.venv/Scripts/python scripts/validate_outputs.py
.venv/Scripts/pragmatiq tokenize data/processed --out outputs/pragmatiq/tok --config configs/pragmatiq_tokenizer.yaml --n-workers 4
```

`--now 2028-10-13` matters. The data runs to 2028-09-30, and intake drops
"future" events by default, so without a pinned clock it would discard every
event after today's date.

Tests (after building the fixture from `data/processed`):

```bash
.venv/Scripts/python scripts/build_test_fixture.py
cd pragmatiq && ../.venv/Scripts/python -m pytest tests -m "not gpu"
```

## Labels

Built by `scripts/build_labels.py` from real outcomes, with eval point 2027-10-01
(24 months of history before it, 12 months of outcome after). pragmatiq cuts
each history at `eval_ts`, so the outcome window never reaches the model.

| task | positive when | population |
| --- | --- | --- |
| `default_12m` | customer enters collections within 12 months | open customers not already in collections |
| `fraud_12m` | customer gets a fraud alert within 12 months | open customers |
| `churn_6m` | relationship closes within 6 months | open customers |
