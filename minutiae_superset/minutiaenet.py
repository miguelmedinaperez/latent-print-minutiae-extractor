"""Full MinutiaeNet (CoarseNet + FineNet) minutiae extraction in PyTorch.

Faithful reproduction of inference() from MinutiaeNet: CoarseNet FCN -> seg
post-processing -> label2mnt -> dual NMS + adaptive threshold -> fuse_nms ->
FineNet soft-fusion -> STFT orientation refinement. Verified to reproduce the TF1
.mnt output on SD27 (exact positions; scores within ~6e-3, angles within ~0.03 rad).
"""
from __future__ import annotations

import numpy as np
import cv2
import torch

from .coarsenet import CoarseNet, label2mnt
from . import postproc as P


@torch.no_grad()
def _coarse_maps(model, img_float_2d, device):
    h, w = img_float_2d.shape
    t = torch.tensor(img_float_2d, dtype=torch.float32).view(1, 1, h, w).to(device)
    out = model(t)
    seg = out["seg_out"].squeeze().cpu().numpy()
    mnt_s = out["mnt_s_out"].squeeze().cpu().numpy()
    mnt_w = out["mnt_w_out"].squeeze(0).permute(1, 2, 0).cpu().numpy()
    mnt_h = out["mnt_h_out"].squeeze(0).permute(1, 2, 0).cpu().numpy()
    mnt_o = out["mnt_o_out"].squeeze(0).permute(1, 2, 0).cpu().numpy()
    return seg, mnt_s, mnt_w, mnt_h, mnt_o


@torch.no_grad()
def _finenet_probs(fine_model, original_image, mnt_nms, device, radio=22):
    patches, idx = [], []
    for i in range(mnt_nms.shape[0]):
        x_begin = int(mnt_nms[i, 1]) - radio
        y_begin = int(mnt_nms[i, 0]) - radio
        patch = original_image[x_begin:x_begin + 2 * radio, y_begin:y_begin + 2 * radio]
        if patch.shape[0] == 0 or patch.shape[1] == 0 or x_begin < 0 or y_begin < 0:
            continue
        p = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_NEAREST)
        patches.append(np.stack([p, p, p], 0).astype(np.float32))
        idx.append(i)
    probs = [None] * mnt_nms.shape[0]
    if patches:
        X = torch.tensor(np.stack(patches)).to(device)
        pm = torch.softmax(fine_model(X), 1)[:, 0].cpu().numpy()
        for j, i in enumerate(idx):
            probs[i] = float(pm[j])
    return probs


def extract(coarse_model, fine_model, img_gray_uint8, device="cuda", use_finenet=True,
            refine_orientation=True, gpu_orientation=False, max_minu=20, min_minu=6):
    """MinutiaeNet minutiae for one HxW uint8 image -> (N,4) x,y,angle,score.

    refine_orientation: if True (faithful), replace each minutia's angle with the STFT
    ridge-flow direction (fuse_minu_orientation). This STFT step is ~80% of the CPU
    post-processing and ONLY affects the angle — set False to keep the network's
    orientation-head angle and run ~4-5x faster (positions/scores/detections identical).
    gpu_orientation: if True, compute the STFT dir_map on the GPU (torch.fft, ~10x
    faster than the numpy STFT); angles match the CPU STFT on unambiguous blocks but a
    few near-tie blocks may differ. Positions/scores are unaffected either way.
    """
    img_size = np.array(img_gray_uint8.shape, dtype=np.int32) // 8 * 8
    image = img_gray_uint8[:img_size[0], :img_size[1]]
    mask = np.ones((img_size[0], img_size[1]))
    original_image = image.copy()

    dir_map = None
    if refine_orientation:
        if gpu_orientation:
            from .gpu_orientation import fast_enhance_texture, get_maps_STFT_gpu
            tex = fast_enhance_texture(
                torch.as_tensor(image.astype(np.float64), device=device)).cpu().numpy()
            dir_map = get_maps_STFT_gpu(tex, device, 64, 16, preprocess=True)
        else:
            texture_img = P.FastEnhanceTexture(image, sigma=2.5, show=False)
            dir_map, _ = P.get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)

    image = image * mask
    seg_out, mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out = _coarse_maps(coarse_model, image, device)

    round_seg = np.round(np.squeeze(seg_out))
    seg_out = 1 - round_seg
    seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    seg_out = cv2.dilate(seg_out, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    max_num_minu, min_num_minu, early_minutiae_thres = max_minu, min_minu, 0.5
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

    if use_finenet and fine_model is not None and mnt_nms.shape[0]:
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

    if refine_orientation and dir_map is not None:
        P.fuse_minu_orientation(dir_map, mnt_nms, mode=3)
    return mnt_nms
