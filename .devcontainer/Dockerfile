# Dev container image — full python:3.12 (more tooling than the slim runtime image), with the system
# libs OpenCV/matplotlib need. Python deps are installed by devcontainer.json's postCreateCommand
# (requirements-dev.txt) so they stay in sync with the repo without rebuilding the image.
FROM python:3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspaces/latent-print-minutiae-extractor
