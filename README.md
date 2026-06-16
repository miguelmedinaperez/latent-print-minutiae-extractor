# latent-print-minutiae-extractor

**Minutiae extraction from latent fingerprints *and* palmprints at 500 dpi, with one fine-tuned
universal model.** A PyTorch port of PyFing's **LEADER** minutiae CNN plus a fine-tuned *universal*
model that is the **best detector on 3 of 4 benchmarks** (held-out, subject-disjoint 5-fold CV) —
see [RESULTS.md](RESULTS.md).

One model handles both print types; the weights (~8 MB) ship in this repo, so it runs out of the box
on CPU or GPU.

> **Built on PyFing's LEADER.** This project is a PyTorch port and a fingerprint/palmprint *fine-tune*
> of **LEADER** (Lightweight End-to-end Attention-gated Dual autoencodER), the minutiae extractor by
> **Raffaele Cappelli** (University of Bologna), distributed in the
> [PyFing](https://github.com/raffaele-cappelli/pyfing) library (© 2023, MIT). The base architecture
> and pretrained weights are entirely the PyFing authors' work — all credit for them goes there. This
> repo adds only the PyTorch port, the fine-tuning, and the serving layer. Please **cite PyFing /
> LEADER** if you use this (see [Credits & citation](#license--attribution)).

## Install

```bash
git clone <repo-url> && cd latent-print-minutiae-extractor
pip install -r requirements.txt          # pinned, reproducible versions (Python 3.12)
```

Dependencies are **pinned** to the versions verified in CI and the Docker build, so installs are
reproducible. GPU is auto-detected. **For an NVIDIA GPU, install the *same* `torch` version from the
CUDA index that matches your card** — recent cards / Blackwell (sm_120) need a CUDA 13 build:

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130   # older cards: a cu12 index
```

Otherwise it falls back to CPU (the default wheel in `requirements.txt`).

## Three ways to use it

### 1. Command line

```bash
python -m leader.infer latent.png --dpi 500 --quality 0.1 --out minutiae.tsv
python -m leader.infer "prints/*.png" --batch --out-dir out/      # many images
python -m leader.infer latent.png --tta --compile --out minutiae.tsv   # TTA and/or CUDA-graph (GPU)
# TSV columns:  x   y   angle(rad)   quality
```

### 2. Python

```python
import cv2
from leader import MinutiaeExtractor

ex = MinutiaeExtractor()                          # loads the universal model (CPU or GPU)
img = cv2.imread("latent.png", cv2.IMREAD_GRAYSCALE)
minutiae = ex.extract(img, dpi=500, quality=0.1)  # [{'x','y','angle','quality'}, ...]

# many images in one pass:
batch = ex.extract_batch([img1, img2, img3], dpi=500)

# tta     = test-time augmentation (~+0.02 loc AP, esp. palms; ~5× forward cost)
# compile = CUDA-graph compile (~2.5× faster on GPU; ~1 min warmup per input size; no-op on CPU)
ex = MinutiaeExtractor(tta=True, compile=True)     # both default off
```

### 3. Web service (container — scales on a GPU cluster)

```bash
docker compose up --build              # one command: builds + serves on http://localhost:8000 (CPU)
# or, by hand:
docker build -t latent-print-minutiae-extractor .
docker run -p 8000:8000 latent-print-minutiae-extractor    # default image is CPU

# (optional) initialize the extractor — set TTA / CUDA-graph compile once:
curl -X POST http://localhost:8000/configure -H "Content-Type: application/json" -d '{"tta": true, "compile": false}'
# extract:
curl -F file=@latent.png "http://localhost:8000/extract?dpi=500&quality=0.1"
# overlay a minutiae file on its print -> PNG (markers coloured by the confidence scale):
curl -F file=@latent.png -F minutiae=@minutiae.tsv http://localhost:8000/plot -o overlay.png
```

**Endpoints** (interactive docs at `/docs`, OpenAPI at `/openapi.json`):

| method · path | parameters | returns |
|---|---|---|
| `GET /health` | — | `{status, device, tta, compiled, half}` |
| `POST /configure` | JSON body `{tta, compile, half}` | (re)initializes the extractor; returns the config |
| `POST /extract` | `file`; query `dpi`, `quality` | `{image, count, minutiae}` |
| `POST /extract_batch` | `files[]`; query `dpi`, `quality` | `{results: [...]}` |
| `POST /plot` | `file` + `minutiae` (TSV/JSON) | overlay **PNG** (markers on the confidence scale) |

`tta` and `compile` are **construction settings**, so they're set once via **`POST /configure`** (the
HTTP form of `MinutiaeExtractor(tta=, compile=)`) — `compile` is GPU-only with a ~1 min warmup per
input size — while `dpi`/`quality` are per-request. The service is **stateless per request** and
scales horizontally: for multi-pod deployments set the config at startup via the `LEADER_TTA` /
`LEADER_COMPILE` env vars (so every pod is consistent), and use `/configure` for single-instance or
dev overrides. A Kubernetes Deployment + Service + HPA example is in
[`deploy/k8s-deployment.yaml`](deploy/k8s-deployment.yaml).

### 4. Visualize

Overlay the extracted minutiae on the print — a hollow circle + a short direction line per minutia,
**coloured by confidence** (RdYlGn: red = low → green = high) with a colorbar:

```bash
python -m leader.viz latent.png --out overlay.png --quality 0.1
```

(`pip install matplotlib`, included in `requirements.txt`.) The direction line is drawn following the
angle convention below.

## Output format & angle convention

Every interface returns the **same minutiae**. The CLI writes **TSV** (one minutia per line); the
Python API and the web service return the equivalent **JSON** dicts.

| field | TSV col | meaning |
|---|---|---|
| `x`, `y` | 1–2 | integer **pixel coordinates in the input image** — origin top-left, `x` →right, `y` →down. (At `dpi≠500` the model runs on a resampled copy and maps coordinates back to your image's grid.) |
| `angle` | 3 | ridge **direction in radians**, range `(−π, π]`, in LEADER's native convention (see below) |
| `quality` | 4 | detection **confidence in (0, 1]** — the detection-map peak; higher = more confident. The `quality=` argument drops anything below it. |

**Angle convention — important for drawing or matching.** The reported `angle` is in LEADER's native
convention. To draw the direction (or compare against GT in the usual examiner convention) on an
image — where the **y axis points down** — **negate the angle** and use cos/sin:

```python
import numpy as np
a  = -m["angle"]                       # LEADER convention -> image pixel convention
L  = 16                                # line length in pixels
x2 = m["x"] + L * np.cos(a)            # endpoint x
y2 = m["y"] + L * np.sin(a)            # endpoint y  (y increases downward)
# draw a line from (m['x'], m['y']) to (x2, y2)
```

This is exactly what `leader.viz` does. (Equivalently: the direction unit vector is
`(cos(angle), −sin(angle))` in image pixel axes.) If your downstream matcher expects the standard
"angle CCW from +x with y up", pass `−angle`.

## Hardware & runtime

Per-image latency on an NVIDIA RTX 5080 (Blackwell) and on CPU:

| | GPU fp16 (default) | GPU fp16 + compile | CPU |
|---|---|---|---|
| fingerprint (~768×800) | **~13 ms** | **~5 ms** | ~0.9 s |
| palmprint (~850×1750) | **~30 ms** | (per-size compile) | ~2.4 s |

- **GPU strongly recommended** (≈ 75× faster than CPU). The model is tiny — ~0.9 M parameters,
  ~8 MB weights — so any modern GPU and ~4 GB RAM suffice; a large palmprint is the heaviest case.
- **fp16 is on by default on GPU** (`MinutiaeExtractor(half=True)`): ~25 % faster and the detected
  minutiae are unchanged — verified for **both positions (100 % overlap) and angles** (Δ median
  0.04°) vs fp32. Pass `half=False` to disable.
- **CUDA graphs give ~2.5× more** (`MinutiaeExtractor(compile=True)`): `torch.compile`'s
  `reduce-overhead` mode captures the network's launch sequence into a replay graph — 13 ms → ~5 ms,
  **identical minutiae**. The model is launch-bound, so this is the biggest lever. Caveat: it
  recompiles per input *size* (~1 min warmup each), so it pays off for fixed-size / high-volume
  same-size workloads. (fp8/NVFP4 do **not** help: cuDNN has no fp8/fp4 *conv* kernels — those
  formats target matmul/transformers, not this conv U-Net.)
- **CUDA note:** install a PyTorch build matching your GPU. NVIDIA Blackwell (sm_120) needs a CUDA 13
  build of PyTorch; older cards work with standard CUDA 12 wheels. CPU works everywhere (slower).
- **Batching does *not* speed things up — scale out instead.** `extract_batch` exists for
  convenience, but a single full-resolution forward already **saturates the GPU**, so per-image time
  is flat regardless of batch size (measured: ~12 ms/img from batch 1 to 32). This is inherent to
  processing at full input resolution — there is no architecture change that adds batch throughput
  without trading detection accuracy (a smaller/lower-resolution model would). To raise throughput,
  **run more workers/pods** (one model each); the web service + horizontal autoscaling is the
  recommended high-volume deployment.

## Fine-tune on your own data

```bash
python -m leader.finetune --data /path/db1 /path/db2 --out my_leader.pt
```

Each `--data` dir holds grayscale images with matching `.xml` GT minutiae
(`<Minutia X=".." Y=".." Angle=".." />`). Defaults reproduce the universal recipe (head +
refinement encoder+decoder, σ=3 Gaussian heatmap, plain BCE, 512 px crops, 60 epochs).
See [RESULTS.md](RESULTS.md#recipe).

## Verify the port

```bash
python -m leader.leader_torch    # loads the shipped weights + runs a forward pass; prints heads OK
```

(The PyTorch port was verified to ~1e-6 vs the original Keras LEADER during development; this command
is a load + forward-pass sanity check.)

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                                          # 18 CPU-only tests
pytest --cov=leader --cov=service --cov-report=term-missing   # coverage (94%)
```

The suite (in `tests/`) is hardware-independent (runs on CPU) and asserts the API *contract*, not a
specific minutia count: the PyTorch port loads and matches Keras, the NMS decode recovers a known
peak, the extract → pad → de-offset → dpi pipeline is exercised with an injected detection map, the
**CLI** (`leader.infer`) TSV/JSON output, the **visualizer** (`leader.viz`, incl. the angle
convention), a 1-epoch **fine-tune** round-trip, and the **FastAPI** endpoints. Coverage is **94 %**
of the runtime code (`leader/` + `service/`); the only runtime file excluded is `leader/dump_leader.py`,
a build-time Keras→`.npz` weight converter that isn't runtime code (see `.coveragerc`). The service
tests skip automatically if the optional `httpx` dep is missing.

## Develop (dev container)

A [Dev Container](https://containers.dev) is included, so you can get a ready-to-hack environment in
one click:

- **VS Code:** open the repo and run **“Dev Containers: Reopen in Container”** (Dev Containers
  extension), or
- **CLI:** `devcontainer up --workspace-folder .` (`npm i -g @devcontainers/cli`).

It builds [`.devcontainer/Dockerfile`](.devcontainer/Dockerfile) (Python 3.12 + the OpenCV/matplotlib
system libs), installs `requirements-dev.txt`, and forwards port 8000 for the web service. The image
is **CPU** by default; for GPU development, base `.devcontainer/Dockerfile` on a CUDA PyTorch image
and add `"runArgs": ["--gpus", "all"]` to `devcontainer.json` (see the GPU note in the `Dockerfile`).
Inside, `pytest -q` runs the suite and `uvicorn service.app:app --reload` serves the API.

**Debug the service in VS Code.** `.vscode/launch.json` ships ready-to-use configs (Python / debugpy
extension). Open the repo, go to **Run & Debug**, pick **“FastAPI: debug (run service/app.py)”** and
press **F5** — `service/app.py` runs uvicorn in-process (so breakpoints bind) on
`http://127.0.0.1:8000`. There's also a **uvicorn `--reload`** config (hot reload, limited breakpoints)
and a **pytest** config. From a terminal the same entrypoint is `python -m service.app`.

## License & attribution

Apache-2.0 (see [LICENSE](LICENSE)).

**Credits — the base model.** The architecture and pretrained weights are **LEADER** (Lightweight
End-to-end Attention-gated Dual autoencodER) by **Raffaele Cappelli** (University of Bologna),
shipped in the **[PyFing](https://github.com/raffaele-cappelli/pyfing)** library (© 2023, MIT). This
repository only **ports it to PyTorch** and **fine-tunes** it on latent fingerprints and palmprints;
all credit for the underlying detector belongs to the PyFing authors. The MIT notice is retained in
[NOTICE](NOTICE). If you use this work, please cite **PyFing / LEADER** (Cappelli, *LEADER*,
arXiv:2602.15493) alongside this repository.

Benchmark comparisons in [RESULTS.md](RESULTS.md) additionally cite FingerNet and MinutiaeNet
(© 2017 D.-L. Nguyen, MIT) for context only — no code or weights from those projects are included.
