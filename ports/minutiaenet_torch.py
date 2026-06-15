"""Full MinutiaeNet (CoarseNet + FineNet) minutiae extraction in PyTorch.

Faithful reproduction of inference() from CoarseNet_model.py: CoarseNet FCN (torch
port) -> seg post-processing -> label2mnt -> dual NMS + adaptive threshold ->
fuse_nms -> FineNet soft-fusion -> STFT orientation refinement -> minutiae list.

All classical post-processing is the verbatim MIT-licensed MinutiaeNet code
(mn_postproc.py); only the two CNNs are PyTorch ports. Verification-driven: the
final minutiae must reproduce the TF1 .mnt output on SD27.
"""
from __future__ import annotations

import numpy as np
import cv2
import torch

from coarsenet_torch import CoarseNetTorch, label2mnt
from finenet_torch import build_finenet
import mn_postproc as P


@torch.no_grad()
def _coarse_maps(model, img_float_2d, device):
    """Run CoarseNet FCN on a HxW float (0-255) image. Returns squeezed maps."""
    h, w = img_float_2d.shape
    t = torch.tensor(img_float_2d, dtype=torch.float32).view(1, 1, h, w).to(device)
    out = model(t)
    seg = out["seg_out"].squeeze().cpu().numpy()
    mnt_s = out["mnt_s_out"].squeeze().cpu().numpy()                    # (h/8,w/8)
    mnt_w = out["mnt_w_out"].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (.,.,8)
    mnt_h = out["mnt_h_out"].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (.,.,8)
    mnt_o = out["mnt_o_out"].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (.,.,180)
    return seg, mnt_s, mnt_w, mnt_h, mnt_o


@torch.no_grad()
def _finenet_probs(fine_model, original_image, mnt_nms, device, radio=22):
    """Replicate the per-minutia FineNet soft-fusion patch extraction.
    Returns a list aligned with mnt_nms rows: P(minutia) or None (keep original)."""
    patches, idx = [], []
    for i in range(mnt_nms.shape[0]):
        x_begin = int(mnt_nms[i, 1]) - radio        # row (Y)
        y_begin = int(mnt_nms[i, 0]) - radio        # col (X)
        patch = original_image[x_begin:x_begin + 2 * radio, y_begin:y_begin + 2 * radio]
        if patch.shape[0] == 0 or patch.shape[1] == 0 or x_begin < 0 or y_begin < 0:
            continue                                 # matches try/except -> keep original
        p = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_NEAREST)
        patches.append(np.stack([p, p, p], 0).astype(np.float32))
        idx.append(i)
    probs = [None] * mnt_nms.shape[0]
    if patches:
        X = torch.tensor(np.stack(patches)).to(device)
        pm = torch.softmax(fine_model(X), 1)[:, 0].cpu().numpy()   # P(minutia) = class 0
        for j, i in enumerate(idx):
            probs[i] = float(pm[j])
    return probs


def extract_minutiae_mn(coarse_model, fine_model, img_gray_uint8, device="cuda",
                        use_finenet=True):
    """Reproduce MinutiaeNet inference() for one image. Returns (mnt (N,4), img_size)."""
    img_size = np.array(img_gray_uint8.shape, dtype=np.int32) // 8 * 8
    image = img_gray_uint8[:img_size[0], :img_size[1]]
    mask = np.ones((img_size[0], img_size[1]))
    original_image = image.copy()

    texture_img = P.FastEnhanceTexture(image, sigma=2.5, show=False)
    dir_map, _ = P.get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)

    image = image * mask                              # mask=ones -> float copy

    seg_out, mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out = _coarse_maps(
        coarse_model, image, device)

    # seg post-processing (verbatim recipe)
    round_seg = np.round(np.squeeze(seg_out))
    seg_out = 1 - round_seg
    seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    seg_out = cv2.dilate(seg_out, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    max_num_minu, min_num_minu, early_minutiae_thres = 20, 6, 0.5

    mnt = label2mnt(np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)),
                    mnt_w_out, mnt_h_out, mnt_o_out, thresh=0)

    mnt_nms_1 = P.py_cpu_nms(mnt, 0.5)
    mnt_nms_2 = P.nms(mnt)
    if mnt_nms_1.shape[0]:
        mnt_nms_1 = mnt_nms_1[np.argsort(mnt_nms_1[:, 3], kind="stable")][::-1]

    mnt_nms_1_copy, mnt_nms_2_copy = mnt_nms_1.copy(), mnt_nms_2.copy()
    while early_minutiae_thres > 0:
        mnt_nms_1 = mnt_nms_1_copy[mnt_nms_1_copy[:, 3] > early_minutiae_thres, :]
        mnt_nms_2 = mnt_nms_2_copy[mnt_nms_2_copy[:, 3] > early_minutiae_thres, :]
        if mnt_nms_1.shape[0] > max_num_minu or mnt_nms_2.shape[0] > max_num_minu:
            mnt_nms_1 = mnt_nms_1[:max_num_minu, :]
            mnt_nms_2 = mnt_nms_2[:max_num_minu, :]
        if mnt_nms_1.shape[0] > min_num_minu and mnt_nms_2.shape[0] > min_num_minu:
            break
        early_minutiae_thres -= 0.05

    mnt_nms = P.fuse_nms(mnt_nms_1, mnt_nms_2)
    final_thres = early_minutiae_thres - 0.05

    if use_finenet and mnt_nms.shape[0]:
        probs = _finenet_probs(fine_model, original_image, mnt_nms, device)
        refined = []
        for i in range(mnt_nms.shape[0]):
            m = mnt_nms[i, :].copy()
            if probs[i] is not None:
                m[3] = (4 * m[3] + probs[i]) / 5
            refined.append(m)
        mnt_nms = np.array(refined)

    if mnt_nms.shape[0] > 0:
        mnt_nms = mnt_nms[mnt_nms[:, 3] > final_thres, :]

    P.fuse_minu_orientation(dir_map, mnt_nms, mode=3)
    return mnt_nms, img_size


def _read_tf1_mnt(path):
    rows = [l.split() for l in open(path).read().splitlines()[2:] if len(l.split()) >= 4]
    return np.array([[float(c) for c in r[:4]] for r in rows]) if rows else np.zeros((0, 4))


def _match(a, b, dpx=8.0):
    """Greedy position match a->b within dpx; returns matched count."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0
    used = np.zeros(b.shape[0], bool)
    n = 0
    for i in range(a.shape[0]):
        d = np.sqrt(((b[:, :2] - a[i, :2]) ** 2).sum(1))
        d[used] = 1e9
        j = d.argmin()
        if d[j] <= dpx:
            used[j] = True
            n += 1
    return n


if __name__ == "__main__":
    WORK = "."  # set to your working dir (reference conversion code)
    coarse = CoarseNetTorch(f"{WORK}/data/crops/coarsenet_keras_weights.pkl").cuda()
    fine = build_finenet(f"{WORK}/data/crops/finenet_keras_weights.pkl").cuda()
    tf1_dir = f"{WORK}/external/MinutiaeNet/output_CoarseNet/inferenceResults/20260612-154103/SD27/mnt_results"
    img_dir = f"{WORK}/external/MinutiaeNet/Dataset/SD27/img_files"
    names = ["B101X9I_1_1", "B102X0I_1_1", "B104X8I_1_1"]
    print("=== MinutiaeNet torch vs TF1 .mnt (SD27) ===")
    for nm in names:
        img = cv2.imread(f"{img_dir}/{nm}.bmp", cv2.IMREAD_GRAYSCALE)
        mnt, _ = extract_minutiae_mn(coarse, fine, img, use_finenet=True)
        tf1 = _read_tf1_mnt(f"{tf1_dir}/{nm}.mnt")
        m = _match(mnt, tf1, dpx=8.0)
        print(f"\n{nm}: torch {mnt.shape[0]:2d} | TF1 {tf1.shape[0]:2d} | pos-matched {m}")
        order = np.argsort(-mnt[:, 3])[:5] if mnt.shape[0] else []
        for i in order:
            print(f"    torch x{mnt[i,0]:6.1f} y{mnt[i,1]:6.1f} o{mnt[i,2]:.3f} s{mnt[i,3]:.4f}")
        for r in tf1[np.argsort(-tf1[:, 3])][:5]:
            print(f"    TF1   x{r[0]:6.1f} y{r[1]:6.1f} o{r[2]:.3f} s{r[3]:.4f}")
