# latent-print-minutiae-extractor

**Minutiae extraction from latent fingerprints *and* palmprints at 500 dpi, with one fine-tuned
universal model.** A PyTorch port of PyFing's **LEADER** minutiae CNN plus a fine-tuned *universal*
model that is the **best detector on 3 of 4 benchmarks** (held-out, subject-disjoint 5-fold CV) —
see [RESULTS.md](RESULTS.md).

One model handles both print types; the weights (~8 MB) ship in this repo, so it runs out of the box
on CPU or GPU.

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

| | GPU fp16 (default) | GPU fp32 | CPU |
|---|---|---|---|
| fingerprint (~768×800) | **~12 ms** | ~16 ms | ~0.9 s |
| palmprint (~850×1750) | **~30 ms** | ~41 ms | ~2.4 s |

- **GPU strongly recommended** (≈ 75× faster than CPU). The model is tiny — ~0.9 M parameters,
  ~8 MB weights — so any modern GPU and ~4 GB RAM suffice; a large palmprint is the heaviest case.
- **fp16 is on by default on GPU** (`MinutiaeExtractor(half=True)`): ~25 % faster and the detected
  minutiae are unchanged (verified — 100 % position overlap vs fp32). Pass `half=False` to disable.
- **CUDA note:** install a PyTorch build matching your GPU. NVIDIA Blackwell (sm_120) needs a CUDA 13
  build of PyTorch; older cards work with stock CUDA 12 wheels. CPU works everywhere (slower).
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
refinement decoder, σ=3 Gaussian heatmap, plain BCE, 60 epochs). See [RESULTS.md](RESULTS.md#recipe).

## Verify the port

```bash
python -m leader.leader_torch    # loads the model and runs a forward pass (parity vs Keras ~1e-6)
```

## License & attribution

Apache-2.0 (see [LICENSE](LICENSE)). The PyTorch port and fine-tuned weights derive from **PyFing /
LEADER** (© 2023 R. Cappelli, MIT); that MIT notice is retained in [NOTICE](NOTICE). Benchmark
comparisons cite FingerNet and MinutiaeNet (© 2017 D.-L. Nguyen, MIT) for context only.
