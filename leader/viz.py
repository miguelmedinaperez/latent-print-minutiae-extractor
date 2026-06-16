"""Visualize extracted minutiae on a print: a hollow circle + a short direction line per minutia,
coloured by confidence (RdYlGn), with a confidence colorbar.

    python -m leader.viz latent.png --out overlay.png --quality 0.1

LEADER reports angle in its own (-theta) convention; this draws -angle so the direction lines
follow the ridge flow in image coordinates.
"""
import argparse
import io
from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from . import MinutiaeExtractor

CMAP = matplotlib.colormaps["RdYlGn"]


def draw(ax, img, minutiae, title=""):
    ax.imshow(img, cmap="gray"); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    L = max(16.0, 0.018 * max(img.shape)); ms = max(6, int(0.004 * max(img.shape)))
    norm = Normalize(0.0, 1.0)
    for m in minutiae:
        a = -m["angle"]                                   # -> image/identity convention for display
        c = CMAP(norm(m["quality"]))
        ax.plot(m["x"], m["y"], "o", mfc="none", mec=c, ms=ms, mew=1.3)
        ax.plot([m["x"], m["x"] + L * np.cos(a)], [m["y"], m["y"] + L * np.sin(a)], "-", color=c, lw=1.1)
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label("confidence", fontsize=8)


def overlay_png(img, minutiae, title=""):
    """Render the confidence-coloured minutiae overlay (+ colorbar) and return it as PNG bytes.
    Shared by the `leader.viz` CLI and the web service's /plot endpoint."""
    fig, ax = plt.subplots(figsize=(7, 7))
    draw(ax, img, minutiae, title)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--dpi", type=int, default=500); ap.add_argument("--quality", type=float, default=0.1)
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (~+0.02 AP, ~5x cost)")
    a = ap.parse_args()
    img = cv.imread(a.image, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    mns = MinutiaeExtractor(tta=a.tta).extract(img, dpi=a.dpi, quality=a.quality)
    Path(a.out).write_bytes(overlay_png(img, mns, f"{Path(a.image).name} — {len(mns)} minutiae"))
    print(f"{len(mns)} minutiae -> {a.out}")


if __name__ == "__main__":
    main()
