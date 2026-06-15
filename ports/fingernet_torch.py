"""PyTorch port of FingerNet (FCN minutiae detector) with the Keras weights.

Faithful reimplementation of get_main_net (deploy outputs) from
external/FingerNet/src/train_test_deploy.py, loaded by layer name from the dumped
Keras weights (fingernet_dump_weights.py). Verification-driven: extract() must
reproduce the TF1 .mnt output. Conv: bias=True; BN: scale=True (gamma,beta,mean,var,
eps=1e-3); PReLU: per-channel. Gabor enhancement filters are loaded from the weights.

Fully convolutional -> any input size (cropped to a multiple of 8).
"""
from __future__ import annotations

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- weight loader (by Keras layer name) ----------------
class W:
    def __init__(self, path):
        self.d = {l["name"]: l["weights"] for l in pickle.load(open(path, "rb"))}

    def conv(self, name, in_ch, out_ch, k, dilation=1, bias=True):
        w = self.d[name]
        pad = dilation * (k - 1) // 2
        m = nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=bias)
        m.weight.data = torch.tensor(np.transpose(w[0], (3, 2, 0, 1)).copy())
        if bias:
            m.bias.data = torch.tensor(w[1].copy())
        return m

    def bn(self, name, ch):
        gamma, beta, mean, var = self.d[name]
        m = nn.BatchNorm2d(ch, eps=1e-3)
        m.weight.data = torch.tensor(gamma.copy())
        m.bias.data = torch.tensor(beta.copy())
        m.running_mean.data = torch.tensor(mean.copy())
        m.running_var.data = torch.tensor(var.copy())
        return m

    def prelu(self, name, ch):
        m = nn.PReLU(ch)
        m.weight.data = torch.tensor(self.d[name][0].reshape(-1).copy())
        return m


class CBP(nn.Module):
    """conv_bn_prelu: Conv -> BN -> PReLU."""
    def __init__(self, kw, conv_name, bn_name, prelu_name, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.c = kw.conv(conv_name, in_ch, out_ch, k, dilation)
        self.b = kw.bn(bn_name, out_ch)
        self.p = kw.prelu(prelu_name, out_ch)

    def forward(self, x):
        return self.p(self.b(self.c(x)))


def gausslabel_matrix(length=180, stride=2, sigma=3):
    arr = np.arange(stride // 2, length, stride)            # 90 orientations
    gp = np.exp(-0.5 * ((np.arange(length + 1) - length / 2) / sigma) ** 2)  # signal.gaussian(181,3)
    delta = np.abs(arr[:, None] - arr[None, :])
    delta = np.minimum(delta, length - delta).astype(int) + length // 2
    return gp[delta].astype(np.float32)                     # (90,90), symmetric


class FineGabor(nn.Module):
    """Loads a stored 25x25 Gabor filter bank conv (1 -> 90)."""
    def __init__(self, kw, name):
        super().__init__()
        self.c = kw.conv(name, 1, 90, 25, bias=True)

    def forward(self, x):
        return self.c(x)


class FingerNetTorch(nn.Module):
    def __init__(self, weights_pkl):
        super().__init__()
        kw = W(weights_pkl)
        # VGG backbone
        self.c1_1 = CBP(kw, "conv1_1", "bn-1_1", "prelu-1_1", 1, 64, 3)
        self.c1_2 = CBP(kw, "conv1_2", "bn-1_2", "prelu-1_2", 64, 64, 3)
        self.c2_1 = CBP(kw, "conv2_1", "bn-2_1", "prelu-2_1", 64, 128, 3)
        self.c2_2 = CBP(kw, "conv2_2", "bn-2_2", "prelu-2_2", 128, 128, 3)
        self.c3_1 = CBP(kw, "conv3_1", "bn-3_1", "prelu-3_1", 128, 256, 3)
        self.c3_2 = CBP(kw, "conv3_2", "bn-3_2", "prelu-3_2", 256, 256, 3)
        self.c3_3 = CBP(kw, "conv3_3", "bn-3_3", "prelu-3_3", 256, 256, 3)
        # multi-scale ASPP (dil 1/4/8) -> ori(90) + seg(1) heads
        self.s1 = CBP(kw, "conv4_1", "bn-4_1", "prelu-4_1", 256, 256, 3, 1)
        self.s2 = CBP(kw, "atrousconv4_2", "bn-4_2", "prelu-4_2", 256, 256, 3, 4)
        self.s3 = CBP(kw, "atrousconv4_3", "bn-4_3", "prelu-4_3", 256, 256, 3, 8)
        self.ori1 = CBP(kw, "convori_1_1", "bn-ori_1_1", "prelu-ori_1_1", 256, 128, 1)
        self.ori2 = CBP(kw, "convori_2_1", "bn-ori_2_1", "prelu-ori_2_1", 256, 128, 1)
        self.ori3 = CBP(kw, "convori_3_1", "bn-ori_3_1", "prelu-ori_3_1", 256, 128, 1)
        self.ori1b = kw.conv("ori_1_2", 128, 90, 1)
        self.ori2b = kw.conv("ori_2_2", 128, 90, 1)
        self.ori3b = kw.conv("ori_3_2", 128, 90, 1)
        self.seg1 = CBP(kw, "convseg_1_1", "bn-seg_1_1", "prelu-seg_1_1", 256, 128, 1)
        self.seg2 = CBP(kw, "convseg_2_1", "bn-seg_2_1", "prelu-seg_2_1", 256, 128, 1)
        self.seg3 = CBP(kw, "convseg_3_1", "bn-seg_3_1", "prelu-seg_3_1", 256, 128, 1)
        self.seg1b = kw.conv("seg_1_2", 128, 1, 1)
        self.seg2b = kw.conv("seg_2_2", 128, 1, 1)
        self.seg3b = kw.conv("seg_3_2", 128, 1, 1)
        # enhancement (fixed gabor filters from weights)
        self.gabor_real = FineGabor(kw, "enh_img_real_1")
        self.gabor_imag = FineGabor(kw, "enh_img_imag_1")
        self.register_buffer("glabel", torch.tensor(gausslabel_matrix()).view(90, 90, 1, 1).permute(1, 0, 2, 3).contiguous())
        # minutiae heads (input = phase + seg = 2 ch)
        self.m1 = CBP(kw, "convmnt_1_1", "bn-mnt_1_1", "prelu-mnt_1_1", 2, 64, 9)
        self.m2 = CBP(kw, "convmnt_2_1", "bn-mnt_2_1", "prelu-mnt_2_1", 64, 128, 5)
        self.m3 = CBP(kw, "convmnt_3_1", "bn-mnt_3_1", "prelu-mnt_3_1", 128, 256, 3)
        self.mo1 = CBP(kw, "convmnt_o_1_1", "bn-mnt_o_1_1", "prelu-mnt_o_1_1", 346, 256, 1)
        self.mo2 = kw.conv("mnt_o_1_2", 256, 180, 1)
        self.mw1 = CBP(kw, "convmnt_w_1_1", "bn-mnt_w_1_1", "prelu-mnt_w_1_1", 256, 256, 1)
        self.mw2 = kw.conv("mnt_w_1_2", 256, 8, 1)
        self.mh1 = CBP(kw, "convmnt_h_1_1", "bn-mnt_h_1_1", "prelu-mnt_h_1_1", 256, 256, 1)
        self.mh2 = kw.conv("mnt_h_1_2", 256, 8, 1)
        self.ms1 = CBP(kw, "convmnt_s_1_1", "bn-mnt_s_1_1", "prelu-mnt_s_1_1", 256, 256, 1)
        self.ms2 = kw.conv("mnt_s_1_2", 256, 1, 1)
        self.eval()

    @staticmethod
    def _img_norm(x):  # per-image z-score (img_normalization)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        v = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + 1e-12)

    def _ori_peak(self, ori):  # ori_highest_peak + select_max
        g = F.conv2d(ori, self.glabel)                       # (B,90,h,w) circular gaussian smoothing
        g = g / (g.amax(dim=1, keepdim=True) + 1e-7)
        g = torch.where(g > 0.999, g, torch.zeros_like(g))
        return g / (g.sum(dim=1, keepdim=True) + 1e-7)

    def forward(self, img01):                                # img01 in [0,1], (B,1,H,W)
        x = self._img_norm(img01)
        for c in (self.c1_1, self.c1_2):
            x = c(x)
        x = F.max_pool2d(x, 2)
        for c in (self.c2_1, self.c2_2):
            x = c(x)
        x = F.max_pool2d(x, 2)
        for c in (self.c3_1, self.c3_2, self.c3_3):
            x = c(x)
        x = F.max_pool2d(x, 2)                               # /8
        ori = (self.ori1b(self.ori1(self.s1(x))) + self.ori2b(self.ori2(self.s2(x)))
               + self.ori3b(self.ori3(self.s3(x))))
        ori_out = torch.sigmoid(ori)                         # (B,90,h,w)
        seg = (self.seg1b(self.seg1(self.s1(x))) + self.seg2b(self.seg2(self.s2(x)))
               + self.seg3b(self.seg3(self.s3(x))))
        seg_out = torch.sigmoid(seg)                         # (B,1,h,w)
        # enhancement
        fr, fi = self.gabor_real(img01), self.gabor_imag(img01)   # (B,90,H,W)
        up_ori = F.interpolate(self._ori_peak(ori_out), scale_factor=8, mode="nearest")
        up_seg = F.interpolate(F.softsign(seg_out), scale_factor=8, mode="nearest")
        enh_real = (fr * up_ori).sum(dim=1, keepdim=True)
        enh_imag = (fi * up_ori).sum(dim=1, keepdim=True)
        enh = torch.atan2(enh_imag, enh_real)
        mc = torch.cat([enh, up_seg], dim=1)                 # (B,2,H,W)
        mc = F.max_pool2d(self.m1(mc), 2)
        mc = F.max_pool2d(self.m2(mc), 2)
        mc = F.max_pool2d(self.m3(mc), 2)                    # /8
        mnt_o = torch.sigmoid(self.mo2(self.mo1(torch.cat([mc, ori_out], dim=1))))
        mnt_w = torch.sigmoid(self.mw2(self.mw1(mc)))
        mnt_h = torch.sigmoid(self.mh2(self.mh1(mc)))
        mnt_s = torch.sigmoid(self.ms2(self.ms1(mc)))
        return seg_out, mnt_o, mnt_w, mnt_h, mnt_s


# ---------------- post-processing (numpy, ported from utils) ----------------
def label2mnt(mnt_s, mnt_w, mnt_h, mnt_o, thresh=0.5):
    """mnt_s (h,w); mnt_w/h (h,w,8); mnt_o (h,w,180) -> (N,4) x,y,angle,score."""
    ys, xs = np.where(mnt_s > thresh)
    if len(ys) == 0:
        return np.zeros((0, 4))
    w = np.argmax(mnt_w, axis=-1)
    h = np.argmax(mnt_h, axis=-1)
    o = np.argmax(mnt_o, axis=-1)
    out = np.zeros((len(ys), 4))
    out[:, 0] = xs * 8 + w[ys, xs]
    out[:, 1] = ys * 8 + h[ys, xs]
    out[:, 2] = np.mod((o[ys, xs] * 2 - 89.0) / 180.0 * np.pi, 2 * np.pi)  # [0,2pi) like TF1
    out[:, 3] = mnt_s[ys, xs]
    return out


def _angle_delta(a, b, m=2 * np.pi):
    d = np.abs(a - b)
    return np.minimum(d, m - d)


def nms(mnt, d_thr=16, o_thr=np.pi / 6):
    if mnt.shape[0] == 0:
        return mnt
    order = np.argsort(-mnt[:, 3])
    mnt = mnt[order]
    keep = np.ones(len(mnt), bool)
    for i in range(len(mnt)):
        if not keep[i]:
            continue
        dx = mnt[i + 1:, 0] - mnt[i, 0]
        dy = mnt[i + 1:, 1] - mnt[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        od = _angle_delta(mnt[i + 1:, 2], mnt[i, 2])
        keep[i + 1:] &= ~((dist <= d_thr) & (od <= o_thr))
    return mnt[keep]


@torch.no_grad()
def extract(model, img_gray_uint8, device="cuda"):
    """img_gray_uint8: HxW uint8. Returns minutiae (N,4): x,y,angle,score."""
    import cv2
    h, w = img_gray_uint8.shape
    h8, w8 = h // 8 * 8, w // 8 * 8
    img = img_gray_uint8[:h8, :w8].astype(np.float32) / 255.0
    t = torch.tensor(img).view(1, 1, h8, w8).to(device)
    seg, mo, mw, mh, ms = model(t)
    seg = np.round(seg.squeeze().cpu().numpy())
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    ms_np = ms.squeeze().cpu().numpy() * np.round(seg)
    mnt = label2mnt(ms_np,
                    mw.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                    mh.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                    mo.squeeze(0).permute(1, 2, 0).cpu().numpy(), thresh=0.5)
    return nms(mnt)


if __name__ == "__main__":
    import json, os, sys
    WORK = "."  # set to your working dir (reference conversion code)
    model = FingerNetTorch(f"{WORK}/data/crops/fingernet_keras_weights.pkl").cuda()
    import cv2
    # Verify on a few SD27 images against the TF1 .mnt output.
    tf1_dir = f"{WORK}/external/FingerNet/output/20260612-185815/0"
    names = ["B101X9I_1_1", "B102X0I_1_1", "B104X8I_1_1"]
    for nm in names:
        img = cv2.imread(f"{WORK}/data/SD27/{nm}.bmp", cv2.IMREAD_GRAYSCALE)
        mnt = extract(model, img)
        tf1 = [l.split() for l in open(f"{tf1_dir}/{nm}.mnt").read().splitlines()[2:] if len(l.split()) >= 3]
        print(f"{nm}: torch {len(mnt)} minutiae | TF1 {len(tf1)}")
        if len(mnt):
            print(f"    torch sample: x{mnt[0,0]:.0f} y{mnt[0,1]:.0f} o{mnt[0,2]:.3f} s{mnt[0,3]:.3f}")
        if tf1:
            print(f"    TF1   sample: {tf1[0]}")
