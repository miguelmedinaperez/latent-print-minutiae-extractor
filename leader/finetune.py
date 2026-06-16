"""Fine-tune the universal LEADER recipe on your own latent prints (the recipe found by the
autoresearch sweep: head + refinement encoder+decoder, Gaussian heatmap sigma=3, plain BCE,
512px crops, 60 epochs).

    python leader/finetune.py --data /path/to/db1 /path/to/db2 --out my_leader.pt

Each --data dir holds grayscale images (.bmp/.png/.jpg) with a matching .xml of GT minutiae
(<Minutia X=".." Y=".." Angle=".." .../>; decimal-comma tolerated). Trains from the shipped PyFing
LEADER weights and writes a fine-tuned state_dict to --out. Pool several DBs by passing several
--data dirs (subject-disjoint CV / evaluation is left to your own protocol)."""
import argparse, glob, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, cv2 as cv, torch
from .leader_torch import LeaderTorch

HERE = Path(__file__).resolve().parent
CROP = 512                       # training crop size (round-004: 512 beats 320 under CV); see --crop
EXTS = ("*.bmp", "*.png", "*.jpg", "*.tif")


def read_gt(xmlpath):
    root = ET.parse(xmlpath).getroot()
    pts = [(float(m.attrib["X"].replace(",", ".")), float(m.attrib["Y"].replace(",", ".")))
           for m in root.findall(".//Minutia")]
    return np.array(pts, np.float32) if pts else np.zeros((0, 2), np.float32)


def heatmap(pts, h, w, sigma):
    H = np.zeros((h, w), np.float32); r = int(3 * sigma)
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        x0, x1 = max(0, ix - r), min(w, ix + r + 1); y0, y1 = max(0, iy - r), min(h, iy + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        H[y0:y1, x0:x1] = np.maximum(H[y0:y1, x0:x1], g)
    return H


def sample_crop(img, pts):
    h0, w0 = img.shape
    if h0 < CROP or w0 < CROP:
        img = cv.copyMakeBorder(img, 0, max(0, CROP - h0), 0, max(0, CROP - w0), cv.BORDER_CONSTANT, value=int(img[0, 0]))
        h0, w0 = img.shape
    if len(pts) and np.random.rand() < 0.7:
        cx, cy = pts[np.random.randint(len(pts))]
        ox = int(np.clip(cx - CROP / 2 + np.random.randint(-40, 41), 0, w0 - CROP))
        oy = int(np.clip(cy - CROP / 2 + np.random.randint(-40, 41), 0, h0 - CROP))
    else:
        ox, oy = np.random.randint(0, w0 - CROP + 1), np.random.randint(0, h0 - CROP + 1)
    crop = img[oy:oy + CROP, ox:ox + CROP]; p = pts - [ox, oy] if len(pts) else pts
    if len(p):
        p = p[(p[:, 0] >= 0) & (p[:, 0] < CROP) & (p[:, 1] >= 0) & (p[:, 1] < CROP)]
    if np.random.rand() < 0.5:
        crop = crop[:, ::-1]; p = np.column_stack([CROP - 1 - p[:, 0], p[:, 1]]) if len(p) else p
    if np.random.rand() < 0.5:
        crop = crop[::-1]; p = np.column_stack([p[:, 0], CROP - 1 - p[:, 1]]) if len(p) else p
    for _ in range(np.random.randint(4)):
        crop = np.rot90(crop)
        if len(p): p = np.column_stack([p[:, 1], CROP - 1 - p[:, 0]])
    return np.ascontiguousarray(crop), (np.ascontiguousarray(p) if len(p) else np.zeros((0, 2), np.float32))


def main():
    global CROP                                   # sample_crop()/heatmap() read this module global
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="dirs of images + matching .xml GT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--unfreeze", default="refine", choices=["head", "headdec", "refine"])
    ap.add_argument("--sigma", type=float, default=3.0); ap.add_argument("--wpos", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=CROP, help="training crop size (recipe default 512)")
    a = ap.parse_args()
    CROP = a.crop
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(a.seed); torch.manual_seed(a.seed)

    W = HERE / "weights"
    model = LeaderTorch(str(W / "leader_weights.npz"), str(W / "leader_layers.json")).to(dev)
    pref = ("head_0", "head_conv_pos")
    if a.unfreeze in ("headdec", "refine"):
        pref += ("dec1_0", "dec1_1", "dec1_2", "dec1_3")
    if a.unfreeze == "refine":
        pref += ("enc1_0", "enc1_1", "enc1_2", "enc1_3")
    for name, mod in model.L.items():
        tr = any(name.startswith(p) for p in pref)
        for p in mod.parameters():
            p.requires_grad = tr
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)

    paths = []
    for d in a.data:
        for e in EXTS:
            paths += glob.glob(str(Path(d) / e))
    data = [(cv.imread(p, cv.IMREAD_GRAYSCALE), read_gt(str(Path(p).with_suffix(".xml")))) for p in sorted(paths)]
    data = [(im, gt) for im, gt in data if im is not None]
    print(f"train images: {len(data)} | trainable params: {sum(p.numel() for p in params):,} | {dev}", flush=True)

    model.train()
    for ep in range(1, a.epochs + 1):
        order = np.random.permutation(len(data)); tot = 0.0; nb = 0; t0 = time.time()
        for bi in range(0, len(order), a.batch):
            xs, ys = [], []
            for k in order[bi:bi + a.batch]:
                img, pts = data[k]; crop, p = sample_crop(img.astype(np.float32), pts)
                xs.append(crop[None]); ys.append(heatmap(p, CROP, CROP, a.sigma)[None])
            x = torch.tensor(np.stack(xs), dtype=torch.float32, device=dev)
            y = torch.tensor(np.stack(ys), dtype=torch.float32, device=dev)
            pos, _, _ = model(x); pos = pos.clamp(1e-6, 1 - 1e-6)
            loss = -(a.wpos * y * torch.log(pos) + (1 - y) * torch.log(1 - pos)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"epoch {ep:3d}  loss {tot/nb:8.4f}  ({time.time()-t0:.1f}s)", flush=True)
    torch.save(model.state_dict(), a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
