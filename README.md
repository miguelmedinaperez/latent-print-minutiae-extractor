# Latent Minutiae Extraction (PyTorch)

GPU PyTorch tooling for **minutiae extraction from latent fingerprints *and* palmprints** at 500 dpi,
plus a **fine-tuned universal LEADER** model that is the strongest single detector across our
benchmarks.

Two parts:

1. **`leader/`** — a verified PyTorch port of PyFing's **LEADER** minutiae CNN, a **fine-tuned
   universal model** (`leader/weights/leader_universal_deploy.pt`, one model for both fingerprints
   and palms), an inference CLI, and a config-driven fine-tune script.
2. **`minutiae_superset/`** — a confidence-ranked **superset** extractor that fuses ported
   **FingerNet** + **MinutiaeNet** (CoarseNet + FineNet), works on any image size at 500 dpi.
   (`ports/` holds the reference Keras→PyTorch port code for those nets.)

## Quickstart — universal LEADER

```bash
pip install torch numpy opencv-python
python leader/infer.py path/to/latent.png --dpi 500 --quality 0.1 --out minutiae.tsv
# -> x  y  angle(rad)  quality   (one row per minutia)
```

The LEADER weights ship in `leader/weights/` (~8 MB total), so this runs out of the box on CPU or
GPU. Verify the port reproduces the original Keras model:

```bash
python leader/leader_torch.py        # prints head parity max|Δ| ~1e-6 vs Keras
```

Fine-tune the universal recipe on your own labelled latents:

```bash
python leader/finetune.py --data /path/db1 /path/db2 --out my_leader.pt
```

## Results

A single fine-tuned **universal LEADER** is the **best detector on 3 of 4 evaluation sets**
(held-out, subject-disjoint 5-fold CV). See **[RESULTS.md](RESULTS.md)** for the full per-dataset
tables vs stock FingerNet, MinutiaeNet, PyFing, and the FingerNet+MinutiaeNet Superset.

## The Superset extractor (FingerNet + MinutiaeNet)

`minutiae_superset/` runs FingerNet + MinutiaeNet and returns their confidence-ranked union; it
tiles arbitrarily-large inputs and resamples to 500 dpi. Its model weights (FingerNet ~19 MB,
CoarseNet ~80 MB, FineNet ~218 MB) are **too large to ship in-repo** — download them from the
release page and place them in `minutiae_superset/weights/`. Then:

```python
from minutiae_superset import MinutiaeExtractor
ex = MinutiaeExtractor(device="cuda")
minutiae = ex.extract(image, dpi=500, method="superset")   # or "fingernet" / "minutiaenet"
```

## Attribution & license

PyTorch ports and redistributed/fine-tuned weights derive from three MIT-licensed projects —
**FingerNet**, **MinutiaeNet** (© 2017 Dinh-Luan Nguyen), and **PyFing / LEADER** (© 2023 R.
Cappelli). See [NOTICE](NOTICE) and [LICENSE](LICENSE). This repository is released under MIT.
