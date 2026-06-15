# CPU image by default (works everywhere). For GPU, see the note at the bottom.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY leader/ ./leader/
COPY service/ ./service/
EXPOSE 8000
# One model per worker; scale by running more pods (k8s) rather than many workers per pod.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# --- GPU build ---------------------------------------------------------------
# The app auto-uses CUDA when available. For a GPU image, base on a CUDA-enabled
# PyTorch (e.g. FROM pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime) and skip the torch
# line in requirements; for NVIDIA Blackwell (sm_120) use a torch built with CUDA 13.
# Run with:  docker run --gpus all -p 8000:8000 <image>
