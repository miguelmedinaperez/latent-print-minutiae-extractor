# Results — fine-tuned universal LEADER on latent fingerprints & palmprints

Held-out, **subject-disjoint 5-fold cross-validation**. The fine-tuned models are evaluated only on
images whose subjects were held out of training; the baseline detectors trained on none of these
images, so the comparison is fair.

**Metrics.** **AP** = area under the confidence-ranked per-image precision–recall curve;
**max-F1** = best single-threshold F1. Both under **loc** (match ≤ 15 px) and **loc+angle**
(≤ 15 px AND ≤ 30°). Columns: **loc AP · loc max-F1 · loc+angle AP · loc+angle max-F1**.
Per-column best is **bold**.

**Datasets.** *SD27* (258) and *LPIDB* (380) are public latent benchmarks; *Internal palm set* (73)
and *Internal fingerprint set* (284) are proprietary in-house latent collections.

> "PyFing" is the **original** LEADER model; "fine-tuned LEADER" is **one universal** model (this repo)
> evaluated per dataset.

### Palmprints — Internal palm set (73)

| method | loc AP | loc F1 | +ang AP | +ang F1 |
|---|---|---|---|---|
| FingerNet | 0.570 | 0.685 | 0.562 | 0.681 |
| MinutiaeNet | 0.164 | 0.374 | 0.059 | 0.220 |
| PyFing (LEADER) | 0.559 | 0.665 | 0.545 | 0.661 |
| **fine-tuned LEADER (universal)** | **0.645** | **0.689** | **0.634** | **0.686** |

### Palmprints — LPIDB (380)

| method | loc AP | loc F1 | +ang AP | +ang F1 |
|---|---|---|---|---|
| FingerNet | 0.607 | 0.698 | 0.600 | 0.694 |
| MinutiaeNet | 0.202 | 0.400 | 0.076 | 0.233 |
| PyFing (LEADER) | 0.680 | 0.732 | 0.669 | 0.729 |
| **fine-tuned LEADER (universal)** | **0.718** | **0.750** | **0.710** | **0.747** |

### Latent fingerprints — SD27 (258)

| method | loc AP | loc F1 | +ang AP | +ang F1 |
|---|---|---|---|---|
| FingerNet | 0.416 | 0.633 | 0.406 | 0.624 |
| **MinutiaeNet** | **0.574** | **0.759** | 0.431 | **0.652** |
| PyFing (LEADER) | 0.174 | 0.513 | 0.163 | 0.508 |
| fine-tuned LEADER (universal) | 0.497 | 0.610 | **0.476** | 0.600 |

### Latent fingerprints — Internal fingerprint set (284)

| method | loc AP | loc F1 | +ang AP | +ang F1 |
|---|---|---|---|---|
| FingerNet | 0.581 | 0.723 | 0.570 | 0.714 |
| MinutiaeNet | 0.305 | 0.526 | 0.114 | 0.305 |
| PyFing (LEADER) | 0.681 | 0.745 | 0.664 | 0.739 |
| **fine-tuned LEADER (universal)** | **0.723** | **0.765** | **0.707** | **0.759** |

## Takeaways

- **The fine-tuned universal LEADER is the best detector on 3 of 4 datasets** — Internal palm set
  (0.645), LPIDB (0.718), Internal fingerprint set (0.723) — with **one model** for both
  fingerprints and palms.
- It transforms LEADER on **SD27** (original 0.174 → 0.497 loc AP, +0.32); there it is 2nd behind
  **MinutiaeNet** (0.574) on detection.
- It pools fingerprints + palms **cleanly** — the universal model matches per-domain specialists
  (no dilution).

## Recipe

Fine-tune LEADER's head + refinement decoder; target = Gaussian heatmap (σ=3) at GT minutiae;
loss = plain BCE on the detection map; 320 px crops + flips/90°-rotations; Adam 1e-4, 60 epochs.
Reproduce with `python -m leader.finetune`. Search findings: positive-weighting hurts (plain BCE
wins), σ=3 is optimal, the decoder must be unfrozen, focal loss underperforms.
