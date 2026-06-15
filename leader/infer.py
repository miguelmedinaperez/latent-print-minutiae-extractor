"""Extract minutiae from a latent print (fingerprint OR palmprint) with the fine-tuned universal
LEADER model. Self-contained: loads the PyTorch LEADER port + the universal weights shipped in
./weights/, runs on CPU or GPU.

    python leader/infer.py path/to/print.png --dpi 500 --quality 0.1 --out minutiae.tsv

Output TSV columns: x  y  angle(rad)  quality. Angle follows LEADER's convention (atan2 of the
cos/sin head); negate it if your ground truth uses the opposite sign.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from leader_torch import LeaderTorch

HERE = Path(__file__).resolve().parent


def load_model(weights_dir=HERE / "weights", finetuned="leader_universal_deploy.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = LeaderTorch(str(weights_dir / "leader_weights.npz"), str(weights_dir / "leader_layers.json")).to(device).eval()
    ft = weights_dir / finetuned
    if ft.exists():
        m.load_state_dict(torch.load(str(ft), map_location=device))
    return m, device


def pad32(img):
    h, w = img.shape
    H, Wd = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    t, l = (H - h) // 2, (Wd - w) // 2
    return cv.copyMakeBorder(img, t, H - h - t, l, Wd - w - l, cv.BORDER_CONSTANT, value=int(img[0, 0])), t, l


@torch.no_grad()
def extract(model, device, img, dpi=500, quality=0.1):
    if dpi != 500:
        s = 500 / dpi
        img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_CUBIC)
    h, w = img.shape
    pad, t, l = pad32(img.astype(np.float32))
    x = torch.tensor(pad[None, None], dtype=torch.float32, device=device)
    pos, dir2, typ = model(x)
    out = []
    for mx, my, ang, ty, ql in model.extract(pos, dir2, typ, q=quality)[0]:
        mx, my = mx - l, my - t
        if 0 <= mx < w and 0 <= my < h:
            out.append((mx, my, ang, ql))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("--quality", type=float, default=0.1, help="detection threshold (lower = more candidates)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, device = load_model()
    img = cv.imread(a.image, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    mns = extract(model, device, img, a.dpi, a.quality)
    lines = [f"{x}\t{y}\t{ang:.6f}\t{ql:.6f}" for x, y, ang, ql in mns]
    txt = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(txt)
    print(f"{len(mns)} minutiae ({device})" + (f" -> {a.out}" if a.out else ":\n" + txt))


if __name__ == "__main__":
    main()
