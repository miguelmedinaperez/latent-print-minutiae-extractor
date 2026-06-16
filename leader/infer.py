"""CLI: extract minutiae from a latent fingerprint or palmprint.

    python -m leader.infer path/to/print.png --dpi 500 --quality 0.1 --out minutiae.tsv
    python -m leader.infer "dir/*.png" --batch --out-dir results/        # many images, batched

TSV columns: x  y  angle(rad)  quality.
"""
import argparse, glob, json
from pathlib import Path
import cv2 as cv
from . import MinutiaeExtractor


def _tsv(mns):
    return "".join(f"{m['x']}\t{m['y']}\t{m['angle']:.6f}\t{m['quality']:.6f}\n" for m in mns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="image path or glob")
    ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("--quality", type=float, default=0.1, help="detection threshold (lower = more)")
    ap.add_argument("--batch", action="store_true", help="treat the glob as a batch (one forward pass)")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (~+0.02 AP, ~5x cost)")
    ap.add_argument("--out", default="", help="write TSV for a single image")
    ap.add_argument("--out-dir", default="", help="write one TSV per image (for globs)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a count")
    a = ap.parse_args()
    paths = sorted(glob.glob(a.image)) or [a.image]
    ex = MinutiaeExtractor(tta=a.tta)
    imgs = [cv.imread(p, cv.IMREAD_GRAYSCALE) for p in paths]
    if any(im is None for im in imgs):
        raise SystemExit("could not read one or more images")
    results = ex.extract_batch(imgs, a.dpi, a.quality) if (a.batch and len(imgs) > 1) \
        else [ex.extract(im, a.dpi, a.quality) for im in imgs]
    for p, mns in zip(paths, results):
        if a.out_dir:
            Path(a.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(a.out_dir) / (Path(p).stem + ".tsv")).write_text(_tsv(mns))
        elif a.out and len(paths) == 1:
            Path(a.out).write_text(_tsv(mns))
        if a.json:
            print(json.dumps({"image": p, "minutiae": mns}))
        else:
            print(f"{Path(p).name}: {len(mns)} minutiae ({ex.device})")


if __name__ == "__main__":
    main()
