#!/usr/bin/env bash
# Full run on the GPU VM: raw tables -> events/profiles -> labels -> verify -> tokenize
# -> pretrain (GPU) -> embed -> probe.
#
#   nohup bash scripts/vm_train.sh > outputs/vm_train.log 2>&1 &     # survives logout
#   tail -f outputs/vm_train.log                                      # watch it
#
# Re-running is safe: finished stages are skipped (delete their output to redo them),
# and pretraining resumes from its last checkpoint.
#
# Env overrides:
#   MODEL_SIZE       small | medium | large          (default small, ~10M params)
#   RUN_NAME         run folder under outputs/runs/  (default bank_<MODEL_SIZE>)
#   PRETRAIN_CONFIG  auto = 1 epoch sized to the GPU, or a YAML like
#                    pragmatiq/configs/pretrain.yaml for fixed steps   (default auto)
#   N_WORKERS        tokenizer processes             (default: half the CPU cores)
#   NOW              intake clock; must be after the data's last event (default 2028-10-13)
#   EVAL_TS          label eval point                (default 2027-10-01)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_SIZE="${MODEL_SIZE:-small}"
RUN_NAME="${RUN_NAME:-bank_${MODEL_SIZE}}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-auto}"
N_WORKERS="${N_WORKERS:-$(( $(nproc) / 2 ))}"
NOW="${NOW:-2028-10-13}"
EVAL_TS="${EVAL_TS:-2027-10-01}"

RAW=data/raw
PROCESSED=data/processed
TOK=outputs/tok
RUNS=outputs/runs
mkdir -p outputs

# shellcheck disable=SC1091
source .venv/bin/activate
stage() { echo; echo "=== [$(date '+%F %T')] $* ==="; }

stage "0. environment"
pragmatiq info | grep -E '"device"|"torch"|"python"'

if [ ! -f "$PROCESSED/events.parquet" ]; then
  stage "1. intake: check the raw tables against the mapping"
  fluxtra-intake --device auto check "$RAW" --mapping configs/intake_mapping.yaml \
      --now "$NOW" --out outputs/intake_check
  stage "1. intake: raw tables -> events.parquet + profiles.parquet"
  fluxtra-intake --device auto run "$RAW" --mapping configs/intake_mapping.yaml \
      --out "$PROCESSED" --now "$NOW" --work-dir outputs/intake_work
else
  stage "1. intake: skipped ($PROCESSED/events.parquet exists)"
fi

if [ ! -d "$PROCESSED/labels" ]; then
  stage "2. labels from real outcomes"
  python scripts/build_labels.py --raw "$RAW" --out "$PROCESSED" --eval-ts "$EVAL_TS"
else
  stage "2. labels: skipped ($PROCESSED/labels exists)"
fi

stage "3. verify events/profiles against the raw tables (24 checks; stops on failure)"
python scripts/validate_outputs.py --samples 50 --now "$NOW"
pragmatiq validate "$PROCESSED"

if [ ! -f "$TOK/shard_manifest.json" ]; then
  stage "4. tokenize ($N_WORKERS workers)"
  pragmatiq tokenize "$PROCESSED" --out "$TOK" --config configs/pragmatiq_tokenizer.yaml \
      --n-workers "$N_WORKERS"
else
  stage "4. tokenize: skipped ($TOK/shard_manifest.json exists)"
fi

stage "5. pretrain $MODEL_SIZE on the GPU (plan first, then train; resumes if interrupted)"
pragmatiq pretrain "$TOK" --name "$RUN_NAME" --model-size "$MODEL_SIZE" \
    --config "$PRETRAIN_CONFIG" --runs-root "$RUNS" --show-config
pragmatiq pretrain "$TOK" --name "$RUN_NAME" --model-size "$MODEL_SIZE" \
    --config "$PRETRAIN_CONFIG" --runs-root "$RUNS" --resume auto

stage "6. embed every customer"
pragmatiq embed "$TOK" --run "$RUNS/$RUN_NAME" --out "outputs/embeddings_${RUN_NAME}.parquet"

stage "7. probe each label task (ROC-AUC vs raw-count baseline)"
for label in "$PROCESSED"/labels/*.parquet; do
  task=$(basename "$label" .parquet)
  pragmatiq --quiet probe "$TOK" --run "$RUNS/$RUN_NAME" --label "$label" \
      | tee "outputs/probe_${RUN_NAME}_${task}.json"
done

stage "done"
echo "model:       $RUNS/$RUN_NAME   (checkpoints/, metrics.jsonl)"
echo "embeddings:  outputs/embeddings_${RUN_NAME}.parquet"
echo "probes:      outputs/probe_${RUN_NAME}_*.json"
