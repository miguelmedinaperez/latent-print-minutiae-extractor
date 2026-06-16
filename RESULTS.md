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
> evaluated per dataset. "+TTA" enables test-time augmentation: detection map averaged over 4 flips
> (`MinutiaeExtractor(tta=True)`, default off).

**Protocol — full image, no segmentation mask.** All numbers here are computed on the **entire latent
image with no segmentation-mask cropping**, using confidence-ranked AP / max-F1. This matters for
**SD27**: the LEADER paper (Cappelli & Ferrara, [arXiv:2602.15493](https://arxiv.org/abs/2602.15493))
instead **crops each image to the ground-truth-mask ridge bounding box and removes a 14 px border**,
and reports optimal-point F1 — under *that* protocol the original LEADER is the **best** SD27 detector
(F1 0.71, zero-shot). LEADER has **no internal segmentation**, so on a full latent (large crime-scene
background) it emits background false positives that AP penalises heavily, whereas FingerNet and
MinutiaeNet survive because they segment internally. That is the main reason **stock "PyFing (LEADER)"
SD27 loc AP reads 0.174 here** rather than ~0.71. These rows therefore measure the **unmasked,
full-image regime** — what a real deployment sees — and are **not directly comparable** to the paper's
mask-cropped numbers.

### Palmprints — Internal palm set (73)

| method | loc AP | loc F1 | +ang AP | +ang F1 |
|---|---|---|---|---|
| FingerNet | 0.570 | 0.685 | 0.562 | 0.681 |
| MinutiaeNet | 0.164 | 0.374 | 0.059 | 0.220 |
| PyFing (LEADER) | 0.559 | 0.665 | 0.545 | 0.661 |
| fine-tuned LEADER (universal) | 0.695 | 0.713 | 0.689 | 0.711 |
| **fine-tuned LEADER (+ TTA)** | **0.711** | **0.727** | **0.704** | **0.724** |

### Palmprints — LPIDB (380)

| method | loc AP | loc F1 | +ang AP | +ang F1 |
|---|---|---|---|---|
| FingerNet | 0.607 | 0.698 | 0.600 | 0.694 |
| MinutiaeNet | 0.202 | 0.400 | 0.076 | 0.233 |
| PyFing (LEADER) | 0.680 | 0.732 | 0.669 | 0.729 |
| fine-tuned LEADER (universal) | 0.758 | 0.770 | 0.751 | 0.767 |
| **fine-tuned LEADER (+ TTA)** | **0.767** | **0.776** | **0.760** | **0.772** |

### Latent fingerprints — SD27 (258)

| method | loc AP | loc F1 | +ang AP | +ang F1 |
|---|---|---|---|---|
| FingerNet | 0.416 | 0.633 | 0.406 | 0.624 |
| **MinutiaeNet** | **0.574** | **0.759** | 0.431 | **0.652** |
| PyFing (LEADER) | 0.174 | 0.513 | 0.163 | 0.508 |
| fine-tuned LEADER (universal) | 0.538 | 0.627 | 0.516 | 0.615 |
| fine-tuned LEADER (+ TTA) | 0.555 | 0.648 | **0.534** | 0.639 |

### Latent fingerprints — Internal fingerprint set (284)

| method | loc AP | loc F1 | +ang AP | +ang F1 |
|---|---|---|---|---|
| FingerNet | 0.581 | 0.723 | 0.570 | 0.714 |
| MinutiaeNet | 0.305 | 0.526 | 0.114 | 0.305 |
| PyFing (LEADER) | 0.681 | 0.745 | 0.664 | 0.739 |
| fine-tuned LEADER (universal) | 0.728 | 0.768 | 0.713 | 0.762 |
| **fine-tuned LEADER (+ TTA)** | **0.731** | **0.772** | **0.717** | **0.767** |

## Takeaways

- **The fine-tuned universal LEADER is the best detector on 3 of 4 datasets** — Internal palm set
  (0.711 with TTA / 0.695 default), LPIDB (0.767 / 0.758), Internal fingerprint set (0.731 / 0.728)
  — with **one model** for both fingerprints and palms.
- It transforms LEADER on **SD27** (original 0.174 → 0.538 loc AP, +0.36; 0.555 with TTA) — both
  full-image, no segmentation (see **Protocol** above); there it is 2nd behind **MinutiaeNet** (0.574)
  on loc detection, and it has the **best loc+angle AP** (0.534 with TTA).
- It pools fingerprints + palms **cleanly** — the universal model matches per-domain specialists
  (no dilution).
- **TTA** (`MinutiaeExtractor(tta=True)`) averages the detection map over 4 flips at ~5× forward
  cost; default-off preserves the fast ~12 ms path. Gains are largest on SD27 (+0.017) and
  Internal palm (+0.016).

## Recipe

Fine-tune LEADER's head + refinement **encoder+decoder** (enc1 + dec1 layers; head + head_conv_pos);
target = Gaussian heatmap (σ=3) at GT minutiae; loss = plain BCE on the detection map; **512 px
crops** + flips/90°-rotations; Adam 1e-4, 60 epochs. Reproduce with `python -m leader.finetune`.
Search findings: positive-weighting hurts (plain BCE wins), σ=3 is optimal, the decoder must be
unfrozen, focal loss underperforms, **512 px training crops beat 320** (more ridge context per sample
— confirmed under 5-fold CV: +0.016 pooled fingerprint loc AP), and **unfreezing the refinement
encoder adds +0.012 overall AP** (dominant gain on palms: +0.025, where deeper capacity better models
complex palm ridge fields).
