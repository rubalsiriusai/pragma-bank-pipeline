#!/usr/bin/env bash
# Download the raw bank tables from S3 into data/raw/.
#
#   aws configure                 # once: access key, secret, region (stored on the VM only)
#   S3_PATH="s3://<bucket>/<prefix>/" bash scripts/vm_get_data.sh
#
# The S3 prefix must contain the 12 table folders (txn_event/, session/, ...), the
# same layout intake's mapping (configs/intake_mapping.yaml) expects under data/raw/.
#
# Required env:
#   S3_PATH   source prefix; keep it in quotes because it may contain spaces.
#             Passed at run time so no bucket name or account ID lives in the code.
set -euo pipefail
cd "$(dirname "$0")/.."

S3_PATH="${S3_PATH:?set S3_PATH first, e.g. S3_PATH=\"s3://your-bucket/your/prefix/\" bash scripts/vm_get_data.sh}"
DEST="data/raw"
TABLES=(collections_case collections_case_snapshot customer_profile customer_snapshot
        fraud_case fraud_case_snapshot interaction_event merchant product_event
        product_snapshot session txn_event)

command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI v2 not installed" >&2; exit 1; }
aws sts get-caller-identity >/dev/null || { echo "ERROR: AWS credentials not set; run 'aws configure'" >&2; exit 1; }

echo "listing: $S3_PATH"
listing=$(aws s3 ls "$S3_PATH")
echo "$listing"

missing=()
for t in "${TABLES[@]}"; do
  grep -q "PRE $t/" <<<"$listing" || missing+=("$t")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: these table folders are not directly under the S3 path: ${missing[*]}" >&2
  echo "       Set S3_PATH to the prefix that contains txn_event/, session/, ... and re-run." >&2
  exit 1
fi

mkdir -p "$DEST"
aws s3 sync "$S3_PATH" "$DEST/" --only-show-errors

echo
echo "downloaded:"
for t in "${TABLES[@]}"; do
  printf "  %-28s %3s files  %s\n" "$t" "$(ls "$DEST/$t"/*.parquet 2>/dev/null | wc -l)" "$(du -sh "$DEST/$t" | cut -f1)"
done
echo "total: $(du -sh "$DEST" | cut -f1)"
