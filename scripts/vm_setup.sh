#!/usr/bin/env bash
# One-time environment setup on a Linux GPU VM.
#
#   bash scripts/vm_setup.sh
#
# Creates .venv (Python 3.12) with CUDA-enabled torch, installs intake and pragmatiq
# (editable, with the [train] extra), and prints `pragmatiq info` so you can confirm
# "device": "cuda". Run it from the repository root.
#
# Env overrides:
#   TORCH_INDEX   PyTorch wheel index, e.g. https://download.pytorch.org/whl/cu126
#                 (default: chosen from the driver's CUDA version shown by nvidia-smi)
#   PYTHON        Python version for the venv (default 3.12)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-3.12}"

# ---- GPU present?
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Install the NVIDIA driver first (the VM image usually has it)." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---- uv (fast Python + package installer)
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ---- pick the torch wheel index from the highest CUDA version the driver supports
if [ -z "${TORCH_INDEX:-}" ]; then
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  cuda=${cuda:-0}
  if   [ "$cuda" -ge 128 ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu128
  elif [ "$cuda" -ge 126 ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu126
  else
    echo "ERROR: driver supports CUDA ${cuda:0:2}.${cuda:2} — too old for current torch wheels." >&2
    echo "       Update the NVIDIA driver, or set TORCH_INDEX to a matching wheel index." >&2
    exit 1
  fi
fi
echo "torch wheels from: $TORCH_INDEX"

# ---- environment
uv venv --python "$PYTHON" .venv
uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" "torch>=2.6"
uv pip install --python .venv/bin/python -e "pragmatiq[train]" -e intake

echo
echo "=== pragmatiq info (look for \"device\": \"cuda\") ==="
.venv/bin/pragmatiq info
echo
echo "done. Activate with:  source .venv/bin/activate"
