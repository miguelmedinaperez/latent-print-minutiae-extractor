"""FingerNet FCN minutiae detector (PyTorch).

Port of get_main_net (deploy) from FingerNet (github.com/592692070/FingerNet, MIT),
verified to reproduce the TF1 .mnt output. Fully convolutional -> any input size
(cropped to a multiple of 8).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._layers import WSrc, CBP, gausslabel_matrix, FineGabor


class FingerNet(nn.Module):
    def __init__(self, src: WSrc | None = None):
        super().__init__()
        src = src or WSrc()
        self.c1_1 = CBP(src, "conv1_1", "bn-1_1", "prelu-1_1", 1, 64, 3)
        self.c1_2 = CBP(src, "conv1_2", "bn-1_2", "prelu-1_2", 64, 64, 3)
        self.c2_1 = CBP(src, "conv2_1", "bn-2_1", "prelu-2_1", 64, 128, 3)
        self.c2_2 = CBP(src, "conv2_2", "bn-2_2", "prelu-2_2", 128, 128, 3)
        self.c3_1 = CBP(src, "conv3_1", "bn-3_1", "prelu-3_1", 128, 256, 3)
        self.c3_2 = CBP(src, "conv3_2", "bn-3_2", "prelu-3_2", 256, 256, 3)
        self.c3_3 = CBP(src, "conv3_3", "bn-3_3", "prelu-3_3", 256, 256, 3)
        self.s1 = CBP(src, "conv4_1", "bn-4_1", "prelu-4_1", 256, 256, 3, 1)
        self.s2 = CBP(src, "atrousconv4_2", "bn-4_2", "prelu-4_2", 256, 256, 3, 4)
        self.s3 = CBP(src, "atrousconv4_3", "bn-4_3", "prelu-4_3", 256, 256, 3, 8)
        self.ori1 = CBP(src, "convori_1_1", "bn-ori_1_1", "prelu-ori_1_1", 256, 128, 1)
        self.ori2 = CBP(src, "convori_2_1", "bn-ori_2_1", "prelu-ori_2_1", 256, 128, 1)
        self.ori3 = CBP(src, "convori_3_1", "bn-ori_3_1", "prelu-ori_3_1", 256, 128, 1)
        self.ori1b = src.conv("ori_1_2", 128, 90, 1)
        self.ori2b = src.conv("ori_2_2", 128, 90, 1)
        self.ori3b = src.conv("ori_3_2", 128, 90, 1)
        self.seg1 = CBP(src, "convseg_1_1", "bn-seg_1_1", "prelu-seg_1_1", 256, 128, 1)
        self.seg2 = CBP(src, "convseg_2_1", "bn-seg_2_1", "prelu-seg_2_1", 256, 128, 1)
        self.seg3 = CBP(src, "convseg_3_1", "bn-seg_3_1", "prelu-seg_3_1", 256, 128, 1)
        self.seg1b = src.conv("seg_1_2", 128, 1, 1)
        self.seg2b = src.conv("seg_2_2", 128, 1, 1)
        self.seg3b = src.conv("seg_3_2", 128, 1, 1)
        self.gabor_real = FineGabor(src, "enh_img_real_1")
        self.gabor_imag = FineGabor(src, "enh_img_imag_1")
        self.register_buffer("glabel", torch.tensor(gausslabel_matrix())
                             .view(90, 90, 1, 1).permute(1, 0, 2, 3).contiguous())
        self.m1 = CBP(src, "convmnt_1_1", "bn-mnt_1_1", "prelu-mnt_1_1", 2, 64, 9)
        self.m2 = CBP(src, "convmnt_2_1", "bn-mnt_2_1", "prelu-mnt_2_1", 64, 128, 5)
        self.m3 = CBP(src, "convmnt_3_1", "bn-mnt_3_1", "prelu-mnt_3_1", 128, 256, 3)
        self.mo1 = CBP(src, "convmnt_o_1_1", "bn-mnt_o_1_1", "prelu-mnt_o_1_1", 346, 256, 1)
        self.mo2 = src.conv("mnt_o_1_2", 256, 180, 1)
        self.mw1 = CBP(src, "convmnt_w_1_1", "bn-mnt_w_1_1", "prelu-mnt_w_1_1", 256, 256, 1)
        self.mw2 = src.conv("mnt_w_1_2", 256, 8, 1)
        self.mh1 = CBP(src, "convmnt_h_1_1", "bn-mnt_h_1_1", "prelu-mnt_h_1_1", 256, 256, 1)
        self.mh2 = src.conv("mnt_h_1_2", 256, 8, 1)
        self.ms1 = CBP(src, "convmnt_s_1_1", "bn-mnt_s_1_1", "prelu-mnt_s_1_1", 256, 256, 1)
        self.ms2 = src.conv("mnt_s_1_2", 256, 1, 1)
        self.eval()

    @staticmethod
    def _img_norm(x):
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        v = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return (x - m) / torch.sqrt(v + 1e-12)

    def _ori_peak(self, ori):
        g = F.conv2d(ori, self.glabel)
        g = g / (g.amax(dim=1, keepdim=True) + 1e-7)
        g = torch.where(g > 0.999, g, torch.zeros_like(g))
        return g / (g.sum(dim=1, keepdim=True) + 1e-7)

    def forward(self, img01):
        x = self._img_norm(img01)
        for c in (self.c1_1, self.c1_2):
            x = c(x)
        x = F.max_pool2d(x, 2)
        for c in (self.c2_1, self.c2_2):
            x = c(x)
        x = F.max_pool2d(x, 2)
        for c in (self.c3_1, self.c3_2, self.c3_3):
            x = c(x)
        x = F.max_pool2d(x, 2)
        ori = (self.ori1b(self.ori1(self.s1(x))) + self.ori2b(self.ori2(self.s2(x)))
               + self.ori3b(self.ori3(self.s3(x))))
        ori_out = torch.sigmoid(ori)
        seg = (self.seg1b(self.seg1(self.s1(x))) + self.seg2b(self.seg2(self.s2(x)))
               + self.seg3b(self.seg3(self.s3(x))))
        seg_out = torch.sigmoid(seg)
        fr, fi = self.gabor_real(img01), self.gabor_imag(img01)
        up_ori = F.interpolate(self._ori_peak(ori_out), scale_factor=8, mode="nearest")
        up_seg = F.interpolate(F.softsign(seg_out), scale_factor=8, mode="nearest")
        enh_real = (fr * up_ori).sum(dim=1, keepdim=True)
        enh_imag = (fi * up_ori).sum(dim=1, keepdim=True)
        enh = torch.atan2(enh_imag, enh_real)
        mc = torch.cat([enh, up_seg], dim=1)
        mc = F.max_pool2d(self.m1(mc), 2)
        mc = F.max_pool2d(self.m2(mc), 2)
        mc = F.max_pool2d(self.m3(mc), 2)
        mnt_o = torch.sigmoid(self.mo2(self.mo1(torch.cat([mc, ori_out], dim=1))))
        mnt_w = torch.sigmoid(self.mw2(self.mw1(mc)))
        mnt_h = torch.sigmoid(self.mh2(self.mh1(mc)))
        mnt_s = torch.sigmoid(self.ms2(self.ms1(mc)))
        return seg_out, mnt_o, mnt_w, mnt_h, mnt_s


def label2mnt(mnt_s, mnt_w, mnt_h, mnt_o, thresh=0.5):
    """FingerNet label2mnt -> (N,4) x,y,angle,score (angle in [0,2pi))."""
    ys, xs = np.where(mnt_s > thresh)
    if len(ys) == 0:
        return np.zeros((0, 4))
    w = np.argmax(mnt_w, axis=-1)
    h = np.argmax(mnt_h, axis=-1)
    o = np.argmax(mnt_o, axis=-1)
    out = np.zeros((len(ys), 4))
    out[:, 0] = xs * 8 + w[ys, xs]
    out[:, 1] = ys * 8 + h[ys, xs]
    out[:, 2] = np.mod((o[ys, xs] * 2 - 89.0) / 180.0 * np.pi, 2 * np.pi)
    out[:, 3] = mnt_s[ys, xs]
    return out


def _angle_delta(a, b, m=2 * np.pi):
    d = np.abs(a - b)
    return np.minimum(d, m - d)


def nms(mnt, d_thr=16, o_thr=np.pi / 6):
    """Distance + orientation NMS (FingerNet)."""
    if mnt.shape[0] == 0:
        return mnt
    mnt = mnt[np.argsort(-mnt[:, 3])]
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
    """FingerNet minutiae for a HxW uint8 image -> (N,4) x,y,angle,score."""
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
