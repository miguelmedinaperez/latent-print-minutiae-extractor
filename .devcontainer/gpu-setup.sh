#!/usr/bin/env bash
# Auto-enable GPU in the dev container. requirements.txt installs the CPU build of torch (works
# everywhere); if an NVIDIA GPU is visible in the container, swap in the matching CUDA build so
# torch.cuda.is_available() is True. cu130 (stable) supports Blackwell / sm_120; for older cards
# override the index, e.g.  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124  before rebuild.
# No-op on CPU-only hosts.
set -e
IDX="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "GPU detected -> installing CUDA build: torch==2.12.0 from $IDX"
  pip install --no-cache-dir --force-reinstall torch==2.12.0 --index-url "$IDX"
else
  echo "No GPU visible in the container -> keeping the CPU torch build."
  echo "  (If you have an NVIDIA GPU: ensure the container gets it — VS Code passes it via"
  echo "   hostRequirements.gpu; or add  \"runArgs\": [\"--gpus\",\"all\"]  to devcontainer.json — and rebuild.)"
fi
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
