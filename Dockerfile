# CPU image by default (works everywhere). For GPU, see the note at the bottom.
# Python pinned to 3.14.6 (the verified version) for reproducible builds; for a fully bit-exact base,
# pin a digest instead:  FROM python:3.14.6-slim@sha256:<digest>. (3.14 verified end-to-end: torch
# cu130 + the numpy/opencv/matplotlib/scipy wheels all ship cp314; older 3.10–3.13 also work.)
FROM python:3.14.6-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Install torch first from a selectable index so the SAME Dockerfile builds CPU or GPU images.
#   CPU (default): PyPI wheel.   GPU: pass a CUDA index (the cu13 wheels bundle the CUDA runtime, so
#   no CUDA base image is needed) — for Blackwell/sm_120 use cu130, older cards a cu12x index:
#     docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t mnx:gpu .
ARG TORCH_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir torch==2.12.0 --index-url ${TORCH_INDEX_URL}
# Install the project itself (pip install .) so the image also gets the console commands
# (leader-extract / leader-finetune / leader-viz). Deps come from requirements.txt via pyproject's
# dynamic metadata; torch is already satisfied above, so it is not re-fetched. README.md is copied
# because pyproject references it as the long-description.
COPY pyproject.toml requirements.txt README.md ./
COPY leader/ ./leader/
COPY service/ ./service/
RUN pip install --no-cache-dir .
EXPOSE 8000
# One model per worker; scale by running more pods (k8s) rather than many workers per pod.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# --- GPU image ---------------------------------------------------------------
# Build with the CUDA torch index (above) and run with the NVIDIA runtime:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t mnx:gpu .
#   docker run --gpus all -p 8000:8000 mnx:gpu        # GET /health then reports "device":"cuda"
# The app auto-uses CUDA when torch sees it; fp16 + compile become available.
