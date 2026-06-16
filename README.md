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
pip install -r requirements.txt          # torch, numpy, opencv, fastapi, uvicorn
```

GPU is auto-detected. For an NVIDIA GPU install a CUDA build of PyTorch that matches your card
(recent cards / Blackwell sm_120 need a CUDA 13 build); otherwise it falls back to CPU.

## Three ways to use it

### 1. Command line

```bash
python -m leader.infer latent.png --dpi 500 --quality 0.1 --out minutiae.tsv
python -m leader.infer "prints/*.png" --batch --out-dir out/      # many images
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

# test-time augmentation: ~+0.02 loc AP (esp. palms), ~5× forward cost, default off
ex_tta = MinutiaeExtractor(tta=True)
```

### 3. Web service (container — scales on a GPU cluster)

```bash
docker build -t minutiae-extractor .
docker run --gpus all -p 8000:8000 minutiae-extractor      # drop --gpus all for CPU
curl -F file=@latent.png "http://localhost:8000/extract?dpi=500&quality=0.1"
```

`POST /extract` (one image) and `POST /extract_batch` (several) return JSON; `GET /health` reports
readiness and device. The service is **stateless** — each pod loads one model and serves
independently — so it scales horizontally. A Kubernetes Deployment + Service + HPA example
(one GPU per pod, autoscaled on load) is in [`deploy/k8s-deployment.yaml`](deploy/k8s-deployment.yaml).

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
python -m leader.leader_torch    # loads the model and runs a forward pass (parity vs Keras ~1e-6)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                         # CPU-only smoke tests: port loads, extraction is well-formed,
                                  # deterministic, single==batch, and the FastAPI endpoints respond
```

The suite (in `tests/`) is hardware-independent (runs on CPU) and asserts the API *contract*, not a
specific minutia count. The service tests skip automatically if the optional `httpx` dep is missing.

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
