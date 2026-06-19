# latent-print-minutiae-extractor

**Open source minutiae extractor from latent fingerprints *and* palmprints at 500 dpi, with one fine-tuned universal model.** A PyTorch port of PyFing's **LEADER** minutiae CNN plus a fine-tuned *universal* model that is the **best detector on 3 of 4 benchmarks** (held-out, subject-disjoint 5-fold CV) — see [RESULTS.md](RESULTS.md).

One model handles both print types; the weights (~8 MB) ship in this repo, so it runs out of the box on CPU or GPU.

> **Built on PyFing's LEADER.** This project is a PyTorch port and a fingerprint/palmprint *fine-tune*
> of **LEADER** (Lightweight End-to-end Attention-gated Dual autoencodER), the minutiae extractor by
> **Raffaele Cappelli** and **Matteo Ferrara** (University of Bologna), distributed in the
> [PyFing](https://github.com/raffaele-cappelli/pyfing) library (© 2023, MIT). The base architecture
> and pretrained weights are entirely the PyFing authors' work — all credit for them goes there. This
> repo adds the PyTorch port, the fine-tuning, and the serving layer. Please **cite PyFing /
> LEADER** if you use this (see [Credits & citation](#license--attribution)).

## Why this project

I spent two years working in a small forensic laboratory on a very limited budget. In places like that, creativity is the key: you build what you cannot buy, run it on the hardware you already have — an ordinary desktop PC — and you measure success not by benchmark scores but by the cases you can actually solve.

Latent fingerprints and palmprints are among the hardest evidence to process. They are partial, blurred, and laid over noisy backgrounds. **Minutiae extraction** — locating the ridge endings and bifurcations that make a print unique — is the foundation of any latent examination, and the commercial tools that do it well are simply out of reach for many small labs. There, open source is the only realistic path.

This project is my attempt to put a capable, free tool in those labs' hands. It stands on the shoulders of the [PyFing](https://github.com/raffaele-cappelli/pyfing) project — I'm grateful to **Raffaele Cappelli** and **Matteo Ferrara** the PyFing authors both for the quality of their **LEADER** model and for releasing it openly. With the help of [Claude Code](https://claude.com/claude-code) and an automated fine-tuning loop inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch), I fine-tuned LEADER into a single **universal** model for both fingerprints and palmprints that:

- **runs on any consumer hardware** — ~0.9 s per fingerprint / ~2.4 s per palmprint on CPU, and ~13 ms / ~30 ms on a desktop NVIDIA GPU (~5 ms with CUDA-graph compile);
- **runs on NVIDIA Blackwell (sm_120, e.g. RTX 5080)** with no code changes — just install the matching CUDA-13 PyTorch wheel (one `pip install`, see [Install](#install));
- **is highly accurate** — the best detector on 3 of 4 latent benchmarks under held-out, subject-disjoint cross-validation: loc AP **0.731** (fingerprints), **0.711** (palms), **0.767** (public LPIDB) **with TTA** — **0.728 / 0.695 / 0.758** by default (TTA is opt-in, ~5× cost) — ahead of FingerNet, MinutiaeNet and the original PyFing (full tables in [RESULTS.md](RESULTS.md)); and
- **deploys in minutes** as Python, a CLI, or a container — with the weights included in the repo.

If you work in, or build for, a forensic lab where the budget is the real constraint, I hope it saves you some effort.

## How this compares to the original PyFing / LEADER

This repo keeps **LEADER's architecture exactly as the PyFing authors designed it** — every gain below comes from porting, fine-tuning, and packaging the model, **not** from changing the detector itself.

**What this version adds**

- **Fine-tuned for latents.** PyFing's weights are pretrained for general use; we fine-tuned on ~1,000 latent fingerprints + palmprints under held-out, subject-disjoint 5-fold cross-validation. The result is the **best detector on 3 of 4 latent benchmarks** and the **best loc+angle accuracy on all 4** — ahead of FingerNet, MinutiaeNet, and the original LEADER (full tables in [RESULTS.md](RESULTS.md)).
- **PyTorch port — GPU- and Blackwell-ready.** The original LEADER ships as Keras/TensorFlow, which has no NVIDIA Blackwell (sm_120) wheel, so on cards like the RTX 5080 it runs **CPU-only**. We ported it to PyTorch (verified to ~1e-6 against the original), so it runs on GPU out of the box — **~13 ms per print (~5 ms with CUDA-graph compile) vs ~0.9 s on CPU**.
- **One universal model for fingerprints *and* palmprints.** It pools both print types with no dilution (it matches per-domain specialists), so a single ~8 MB model handles both — where the original is one general model not specialised for latent palms.
- **A deployment layer the research library doesn't have** — CLI, Python API, FastAPI web service +   container + Kubernetes example, optional test-time augmentation and CUDA-graph compilation, and a   minutiae visualizer.

**Why our evaluation — and especially SD27 — differs from the LEADER paper**

We report **every number on the full latent image with no segmentation-mask cropping**, ranked by **confidence (AP / max-F1)**, under **held-out, subject-disjoint 5-fold CV** — i.e. what a real deployment actually sees. The LEADER paper (Cappelli & Ferrara, [arXiv:2602.15493](https://arxiv.org/abs/2602.15493)) instead **crops each image to the ground-truth-mask ridge bounding box, drops a 14 px border, and reports optimal-point F1** (zero-shot). The two protocols measure different things and are **not directly comparable**.

That is the whole reason **stock LEADER's SD27 loc AP reads 0.174 here, not the paper's ~0.71**: LEADER has **no internal segmentation**, so on a full crime-scene latent — a small print on a large, noisy background — it fires **background false minutiae** that confidence-ranked AP penalises heavily. The paper's mask-crop removes exactly that background; FingerNet and MinutiaeNet survive the full image only because they segment internally. (We did evaluate adding a segmentation step: it recovers much of that SD27 gap, but it **only helps heavy-background latents and degrades clean, full-frame prints**, so the shipped model is deliberately kept **segmentation-free** — apply your own foreground mask downstream if your latents have heavy background.)

Under this honest full-image protocol our fine-tune still **transforms** SD27 — **0.174 → 0.538 loc AP (0.555 with TTA)** — and gives the **best loc+angle AP** on the set. It simply isn't comparable to the paper's mask-cropped F1, and we don't claim it is.

## Install

```bash
git clone <repo-url> && cd latent-print-minutiae-extractor
pip install -r requirements.txt          # pinned, reproducible versions (Python 3.14)
```

Dependencies are **pinned** to the versions verified by the Docker build and the test suite, so installs are reproducible. GPU is auto-detected. **For an NVIDIA GPU, install the *same* `torch` version from the CUDA index that matches your card** — recent cards / Blackwell (sm_120) need a CUDA 13 build:

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
# TSV: a header row, then  x  y  angle(rad)  quality  type(E/B)  per minutia  (--json for JSON instead)
```

After `pip install .` (or `pip install -e .`) the same commands are also on your `PATH` as
**`leader-extract`**, **`leader-finetune`**, and **`leader-viz`** — e.g. `leader-extract latent.png --out minutiae.tsv`.

### 2. Python

```python
import cv2
from leader import MinutiaeExtractor

ex = MinutiaeExtractor()                          # loads the universal model (CPU or GPU)
img = cv2.imread("latent.png", cv2.IMREAD_GRAYSCALE)
minutiae = ex.extract(img, dpi=500, quality=0.1)  # [{'x','y','angle','quality','type'}, ...]

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
# GPU image (cu130 = Blackwell/sm_120; older cards: a cu12x index), run with the NVIDIA runtime:
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t mnx:gpu .
docker run --gpus all -p 8000:8000 mnx:gpu                 # /health then reports "device":"cuda"

# (optional) initialize the extractor — set TTA / CUDA-graph compile once:
curl -X POST http://localhost:8000/configure -H "Content-Type: application/json" -d '{"tta": true, "compile": false}'
# extract -> the minutiae file (TSV: header row, then x⇥y⇥angle⇥quality⇥type; count in X-Minutiae-Count):
curl -F file=@latent.png "http://localhost:8000/extract?dpi=500&quality=0.1" -o minutiae.tsv
#   ...or JSON instead:  curl -F file=@latent.png "http://localhost:8000/extract?format=json"
# overlay a minutiae file on its print -> PNG (markers coloured by the confidence scale):
curl -F file=@latent.png -F minutiae=@minutiae.tsv http://localhost:8000/plot -o overlay.png
```

**Endpoints** (interactive docs at `/docs`, OpenAPI at `/openapi.json`):

| method · path | parameters | returns |
|---|---|---|
| `GET /health` | — | `{status, device, tta, compiled, half}` |
| `POST /configure` | JSON body `{tta, compile, half}` | (re)initializes the extractor; returns the config |
| `POST /extract` | `file`; query `dpi`, `quality`, `format` | the **minutiae TSV file** (`text/tab-separated-values`, header row + `x⇥y⇥angle⇥quality⇥type` — same as `leader-extract`; count in `X-Minutiae-Count`), or **JSON** with `?format=json` |
| `POST /extract_batch` | `files[]`; query `dpi`, `quality`, `format` | a **ZIP** of one `<name>.tsv` per image, or **JSON** `{results}` with `?format=json` |
| `POST /plot` | `file` + `minutiae` (TSV/JSON) | overlay **PNG** (markers on the confidence scale) |

> **Testing in Swagger (`/docs`).** The single-file endpoints — `/extract`, `/plot` — have a real file
> picker and work interactively. **`/extract_batch` takes a *list* of files, which Swagger UI's "Try it
> out" can't attach** (it renders a text box, not a file picker — a Swagger-UI limitation with file arrays
> under OpenAPI 3.1, not a service bug). Test it with `curl`/Postman/Python instead, repeating the `files`
> field once per image:
> ```bash
> curl -F files=@a.png -F files=@b.png "http://localhost:8000/extract_batch?dpi=500&quality=0.1" -o minutiae.zip
> # or JSON:  curl -F files=@a.png -F files=@b.png "http://localhost:8000/extract_batch?format=json"
> ```

`tta` and `compile` are **construction settings**, so they're set once via **`POST /configure`** (the HTTP form of `MinutiaeExtractor(tta=, compile=)`) — `compile` is GPU-only with a ~1 min warmup per input size — while `dpi`/`quality` are per-request. The service is **stateless per request** and scales horizontally: for multi-pod deployments set the config at startup via the `LEADER_TTA` / `LEADER_COMPILE` env vars (so every pod is consistent), and use `/configure` for single-instance or dev overrides. A Kubernetes Deployment + Service + HPA example is in [`deploy/k8s-deployment.yaml`](deploy/k8s-deployment.yaml).

### 4. Visualize

Overlay the extracted minutiae on the print — a hollow circle + a short direction line per minutia, **coloured by confidence** (RdYlGn: red = low → green = high) with a colorbar:

```bash
python -m leader.viz latent.png --out overlay.png --quality 0.1
```

(`pip install matplotlib`, included in `requirements.txt`.) The direction line is drawn following the angle convention below.

## Output format & angle convention

Every interface returns the **same five fields**. The CLI and `/extract` write **TSV** — a header row, then one minutia per line; the Python API returns the equivalent **JSON** dicts, and `--json` (CLI) / `?format=json` (service) give JSON there too.

| field | TSV col | meaning |
|---|---|---|
| `x`, `y` | 1–2 | integer **pixel coordinates in the input image** — origin top-left, `x` →right, `y` →down. (At `dpi≠500` the model runs on a resampled copy and maps coordinates back to your image's grid.) |
| `angle` | 3 | ridge **direction in radians**, range `(−π, π]`, in LEADER's native convention (see below) |
| `quality` | 4 | detection **confidence in (0, 1]** — the detection-map peak; higher = more confident. The `quality=` argument drops anything below it. |
| `type` | 5 | minutia type — **`E`** (ridge ending) or **`B`** (bifurcation), from LEADER's type head. **Caveat:** fine-tuning supervised detection + direction, not type; the type head is inherited from LEADER's pretraining and not re-validated on latents — treat it as best-effort. |

The TSV's first line is a **header row** (`x⇥y⇥angle⇥quality⇥type`); our parsers (e.g. `/plot`) skip it, and `pandas.read_csv(sep="\t")` picks it up as column names.

**Angle convention — important for drawing or matching.** The reported `angle` is in LEADER's native convention. To draw the direction (or compare against GT in the usual examiner convention) on an image — where the **y axis points down** — **negate the angle** and use cos/sin:

```python
import numpy as np
a  = -m["angle"]                       # LEADER convention -> image pixel convention
L  = 16                                # line length in pixels
x2 = m["x"] + L * np.cos(a)            # endpoint x
y2 = m["y"] + L * np.sin(a)            # endpoint y  (y increases downward)
# draw a line from (m['x'], m['y']) to (x2, y2)
```

This is exactly what `leader.viz` does. (Equivalently: the direction unit vector is `(cos(angle), −sin(angle))` in image pixel axes.) If your downstream matcher expects the standard "angle CCW from +x with y up", pass `−angle`.

## Hardware & runtime

Per-image latency on an NVIDIA RTX 5080 (Blackwell) and on CPU:

| | GPU fp16 (default) | GPU fp16 + compile | CPU |
|---|---|---|---|
| fingerprint (~768×800) | **~13 ms** | **~5 ms** | ~0.9 s |
| palmprint (~850×1750) | **~30 ms** | (per-size compile) | ~2.4 s |

- **GPU strongly recommended** (≈ 75× faster than CPU). The model is tiny — ~0.9 M parameters (~8 MB of weight files on disk: the base + fine-tuned checkpoints) — so any modern GPU and ~4 GB RAM suffice; a large palmprint is the heaviest case.
- **fp16 is on by default on GPU** (`MinutiaeExtractor(half=True)`): ~25 % faster and the detected   minutiae are unchanged — verified for **both positions (100 % overlap) and angles** (Δ median 0.04°) vs fp32. Pass `half=False` to disable.
- **CUDA graphs give ~2.5× more** (`MinutiaeExtractor(compile=True)`): `torch.compile`'s   `reduce-overhead` mode captures the network's launch sequence into a replay graph — 13 ms → ~5 ms,   **identical minutiae**. The model is launch-bound, so this is the biggest lever. Caveat: it recompiles per input *size* (~1 min warmup each), so it pays off for fixed-size / high-volume same-size workloads. (fp8/NVFP4 do **not** help: cuDNN has no fp8/fp4 *conv* kernels — those formats target matmul/transformers, not this conv U-Net.)
- **CUDA note:** install a PyTorch build matching your GPU. NVIDIA Blackwell (sm_120) needs a CUDA 13 build of PyTorch; older cards work with standard CUDA 12 wheels. CPU works everywhere (slower).
- **Batching does *not* speed things up — scale out instead.** `extract_batch` exists for convenience, but a single full-resolution forward already **saturates the GPU**, so per-image time is flat regardless of batch size (measured: ~12 ms/img from batch 1 to 32). This is inherent to processing at full input resolution — there is no architecture change that adds batch throughput without trading detection accuracy (a smaller/lower-resolution model would). To raise throughput, **run more workers/pods** (one model each); the web service + horizontal autoscaling is the recommended high-volume deployment.

## Fine-tune on your own data

```bash
python -m leader.finetune --data /path/db1 /path/db2 --out my_leader.pt
```

Each `--data` dir holds grayscale images with matching `.xml` GT minutiae (`<Minutia X=".." Y=".." Angle=".." />`). Defaults reproduce the universal recipe (head + refinement encoder+decoder, σ=3 Gaussian heatmap, plain BCE, 512 px crops, 60 epochs).
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
pytest -q                                          # 26 CPU-only tests
pytest --cov=leader --cov=service --cov-report=term-missing   # coverage (96%)
```

The suite (in `tests/`) is hardware-independent (runs on CPU) and asserts the API *contract*, not a specific minutia count: the PyTorch port loads and matches Keras, the NMS decode recovers a known peak, the extract → pad → de-offset → dpi pipeline is exercised with an injected detection map, the **CLI** (`leader.infer`) TSV/JSON output, the **visualizer** (`leader.viz`, incl. the angle convention), the **TTA** detection-map averaging (incl. the CUDA-graph buffer-reuse guard), a 1-epoch **fine-tune** round-trip, and the **FastAPI** endpoints. Coverage is **96 %** of the runtime code (`leader/` + `service/`), which is measured in full — the build-time Keras→`.npz` weight converter lives outside the package, in [`tools/dump_leader.py`](tools/dump_leader.py), and isn't runtime code. The service tests skip automatically if the optional `httpx` dep is missing.

## Develop (dev container)

A [Dev Container](https://containers.dev) is included, so you can get a ready-to-hack environment in one click:

- **VS Code:** open the repo and run **“Dev Containers: Reopen in Container”** (Dev Containers extension), or
- **CLI:** `devcontainer up --workspace-folder .` (`npm i -g @devcontainers/cli`).

It builds [`.devcontainer/Dockerfile`](.devcontainer/Dockerfile) (Python 3.14 + the OpenCV/matplotlib system libs), installs `requirements-dev.txt`, and forwards port 8000 for the web service. Inside, `pytest -q` runs the suite and `uvicorn service.app:app --reload` serves the API.

**GPU is automatic.** The container requests the host GPU (`hostRequirements.gpu`), and on first create [`.devcontainer/gpu-setup.sh`](.devcontainer/gpu-setup.sh) detects it and swaps the CPU torch for the matching **CUDA build** (cu130 — supports Blackwell / sm_120; older cards: set `TORCH_INDEX_URL` to a cu12x index). On a CPU-only host it's a no-op. After “Reopen in Container”, `GET /health` should report `"device":"cuda"`; if it still says `cpu`, your tooling didn't pass the GPU — add `"runArgs": ["--gpus", "all"]` to `devcontainer.json` and rebuild. (If `device:cpu` persists, it's one of: no GPU passthrough, or a CPU torch build — `gpu-setup.sh` prints which.)

**Debug the service in VS Code.** `.vscode/launch.json` ships ready-to-use configs (Python / debugpy extension). Open the repo, go to **Run & Debug**, pick **“FastAPI: debug (run service/app.py)”** and press **F5** — `service/app.py` runs uvicorn in-process (so breakpoints bind) on `http://127.0.0.1:8000`. There's also a **uvicorn `--reload`** config (hot reload, limited breakpoints) and a **pytest** config. From a terminal the same entrypoint is `python -m service.app`.

## License & attribution

Apache-2.0 (see [LICENSE](LICENSE)).

**Credits — the base model.** The architecture and pretrained weights are **LEADER** (Lightweight End-to-end Attention-gated Dual autoencodER) by **Raffaele Cappelli** and **Matteo Ferrara**  (University of Bologna), shipped in the **[PyFing](https://github.com/raffaele-cappelli/pyfing)** library (© 2023, MIT). This repository **ports it to PyTorch** and **fine-tunes** it on latent fingerprints and palmprints; all credit for the underlying detector belongs to the PyFing authors. The MIT notice is retained in [NOTICE](NOTICE). If you use this work, please cite **PyFing / LEADER** (Cappelli & Ferrara, *LEADER*, arXiv:2602.15493) alongside this repository.

Benchmark comparisons in [RESULTS.md](RESULTS.md) additionally cite FingerNet and MinutiaeNet (© 2017 D.-L. Nguyen, MIT) for context only — no code or weights from those projects are included.
