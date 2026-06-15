"""PyTorch port of MinutiaeNet CoarseNet (FCN minutiae detector) with Keras weights.

Faithful reimplementation of CoarseNetmodel (deploy outputs) from
external/MinutiaeNet/CoarseNet/CoarseNet_model.py, loaded by layer name from the
dumped Keras weights (coarsenet_dump_weights.py). Verification-driven: the raw FCN
maps must reproduce the TF1 deploy outputs (coarsenet_ref_maps.pkl).

CoarseNet vs FingerNet:
  - deeper backbone with residual add() blocks; first conv is 5x5
  - ASPP level_3/level_4 (dil 4/8) read from conv_block2, level_2 (dil 1) from conv_block3
  - mnt part also has residual blocks (Block3 has a cross-skip pattern)
  - label2mnt negates the angle: (-angle) % 2pi

Shares the verified W / CBP / gausslabel_matrix / FineGabor infrastructure with
fingernet_torch. Fully convolutional -> any input size (cropped to a multiple of 8).
img_normalization is scale-invariant and gabor->atan2 is phase-only, so the network
is fed the raw grayscale (0-255) exactly as MinutiaeNet's inference() does.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fingernet_torch import W, CBP, gausslabel_matrix, FineGabor


class CoarseNetTorch(nn.Module):
    def __init__(self, weights_pkl):
        super().__init__()
        kw = W(weights_pkl)
        self.cbp = nn.ModuleDict()
        self.proj = nn.ModuleDict()

        def cbp(name, in_ch, out_ch, k, dil=1):
            cn = ("atrousconv" if dil > 1 else "conv") + name
            self.cbp[name] = CBP(kw, cn, "bn-" + name, "prelu-" + name,
                                 in_ch, out_ch, k, dil)

        def proj(name, in_ch, out_ch):
            self.proj[name] = kw.conv(name, in_ch, out_ch, 1)

        # ---- backbone -----------------------------------------------------
        cbp("1_0", 1, 64, 5)
        cbp("1_1", 64, 64, 3)
        cbp("1_2", 64, 64, 3)
        # Block 1 (128ch, 3 residual units) -> conv_block1
        for s in ("", "b", "c"):
            cbp(f"2_1{s}", 128 if s else 64, 128, 3)
            cbp(f"2_2{s}", 128, 128, 3)
            cbp(f"2_3{s}", 128, 128, 3)
        # Block 2 (256ch, 2 residual units) -> conv_block2
        for j, s in enumerate(("", "b")):
            cbp(f"3_1{s}", 128 if j == 0 else 256, 256, 3)
            cbp(f"3_2{s}", 256, 256, 3)
            cbp(f"3_3{s}", 256, 256, 3)
        # Block 3 (512ch, 1 residual unit) + projection -> conv_block3
        cbp("3_1c", 256, 512, 3)
        cbp("3_2c", 512, 512, 3)
        cbp("3_3c", 512, 512, 3)
        cbp("3_4c", 512, 256, 3)

        # ---- multi-scale ASPP --------------------------------------------
        cbp("4_1", 256, 256, 3, 1)   # from conv_block3
        cbp("4_2", 256, 256, 3, 4)   # from conv_block2
        cbp("4_3", 256, 256, 3, 8)   # from conv_block2
        for lvl in ("1", "2", "3"):
            cbp(f"ori_{lvl}_1", 256, 128, 1)
            proj(f"ori_{lvl}_2", 128, 90)
            cbp(f"seg_{lvl}_1", 256, 128, 1)
            proj(f"seg_{lvl}_2", 128, 1)

        # ---- enhancement (fixed gabor filters from weights) --------------
        self.gabor_real = FineGabor(kw, "enh_img_real_1")
        self.gabor_imag = FineGabor(kw, "enh_img_imag_1")
        self.register_buffer(
            "glabel",
            torch.tensor(gausslabel_matrix()).view(90, 90, 1, 1).permute(1, 0, 2, 3).contiguous())

        # ---- mnt part (residual) -----------------------------------------
        # Block 1 (64ch 9x9, 2 residual units)
        cbp("mnt_1_1", 2, 64, 9)
        cbp("mnt_1_2", 64, 64, 9)
        cbp("mnt_1_3", 64, 64, 9)
        cbp("mnt_1_1b", 64, 64, 9)
        cbp("mnt_1_2b", 64, 64, 9)
        cbp("mnt_1_3b", 64, 64, 9)
        # Block 2 (128ch 5x5, 2 residual units)
        cbp("mnt_2_1", 64, 128, 5)
        cbp("mnt_2_2", 128, 128, 5)
        cbp("mnt_2_3", 128, 128, 5)
        cbp("mnt_2_1b", 128, 128, 5)
        cbp("mnt_2_2b", 128, 128, 5)
        cbp("mnt_2_3b", 128, 128, 5)
        # Block 3 (256ch 3x3, cross-skip)
        cbp("mnt_3_1", 128, 256, 3)
        cbp("mnt_3_2", 256, 256, 3)
        cbp("mnt_3_3", 256, 256, 3)
        cbp("mnt_3_4", 256, 256, 3)
        # mnt heads
        cbp("mnt_o_1_1", 346, 256, 1)   # 256 (mnt_conv) + 90 (ori_out_1)
        proj("mnt_o_1_2", 256, 180)
        cbp("mnt_w_1_1", 256, 256, 1)
        proj("mnt_w_1_2", 256, 8)
        cbp("mnt_h_1_1", 256, 256, 1)
        proj("mnt_h_1_2", 256, 8)
        cbp("mnt_s_1_1", 256, 256, 1)
        proj("mnt_s_1_2", 256, 1)
        self.eval()

    # ---- helpers (identical to FingerNet) --------------------------------
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

    def _res3(self, x, n1, n2, n3):
        c1 = self.cbp[n1](x)
        c = self.cbp[n2](c1)
        c = self.cbp[n3](c)
        return c + c1

    def forward(self, img):
        """img: (B,1,H,W) raw grayscale (0-255). Returns a dict of deploy maps (NCHW)."""
        x = self._img_norm(img)
        for n in ("1_0", "1_1", "1_2"):
            x = self.cbp[n](x)
        x = F.max_pool2d(x, 2)                                   # /2
        x = self._res3(x, "2_1", "2_2", "2_3")
        x = self._res3(x, "2_1b", "2_2b", "2_3b")
        x = self._res3(x, "2_1c", "2_2c", "2_3c")
        conv_block1 = F.max_pool2d(x, 2)                         # /4
        x = self._res3(conv_block1, "3_1", "3_2", "3_3")
        x = self._res3(x, "3_1b", "3_2b", "3_3b")
        conv_block2 = F.max_pool2d(x, 2)                         # /8
        x = self._res3(conv_block2, "3_1c", "3_2c", "3_3c")
        conv_block3 = self.cbp["3_4c"](x)                        # /8

        # ASPP: level_2 <- conv_block3 (dil1); level_3/4 <- conv_block2 (dil4/8)
        level_2 = self.cbp["4_1"](conv_block3)
        level_3 = self.cbp["4_2"](conv_block2)
        level_4 = self.cbp["4_3"](conv_block2)
        ori = (self.proj["ori_1_2"](self.cbp["ori_1_1"](level_2))
               + self.proj["ori_2_2"](self.cbp["ori_2_1"](level_3))
               + self.proj["ori_3_2"](self.cbp["ori_3_1"](level_4)))
        ori_out = torch.sigmoid(ori)                            # (B,90,h,w)
        seg = (self.proj["seg_1_2"](self.cbp["seg_1_1"](level_2))
               + self.proj["seg_2_2"](self.cbp["seg_2_1"](level_3))
               + self.proj["seg_3_2"](self.cbp["seg_3_1"](level_4)))
        seg_out = torch.sigmoid(seg)                            # (B,1,h,w)

        # enhancement
        fr, fi = self.gabor_real(img), self.gabor_imag(img)     # (B,90,H,W)
        up_ori = F.interpolate(self._ori_peak(ori_out), scale_factor=8, mode="nearest")
        up_seg = F.interpolate(F.softsign(seg_out), scale_factor=8, mode="nearest")
        enh_real = (fr * up_ori).sum(dim=1, keepdim=True)
        enh_imag = (fi * up_ori).sum(dim=1, keepdim=True)
        enh = torch.atan2(enh_imag, enh_real)
        mc = torch.cat([enh, up_seg], dim=1)                   # (B,2,H,W)

        # mnt part
        mc = self._res3(mc, "mnt_1_1", "mnt_1_2", "mnt_1_3")
        mc = self._res3(mc, "mnt_1_1b", "mnt_1_2b", "mnt_1_3b")
        mc = F.max_pool2d(mc, 2)                                 # /2
        mc = self._res3(mc, "mnt_2_1", "mnt_2_2", "mnt_2_3")
        mc = self._res3(mc, "mnt_2_1b", "mnt_2_2b", "mnt_2_3b")
        mc = F.max_pool2d(mc, 2)                                 # /4
        m1 = self.cbp["mnt_3_1"](mc)
        m2 = self.cbp["mnt_3_2"](m1)
        m3 = self.cbp["mnt_3_3"](m2) + m1
        m4 = self.cbp["mnt_3_4"](m3) + m2
        mc = F.max_pool2d(m4, 2)                                 # /8

        mnt_o = torch.sigmoid(self.proj["mnt_o_1_2"](
            self.cbp["mnt_o_1_1"](torch.cat([mc, ori_out], dim=1))))
        mnt_w = torch.sigmoid(self.proj["mnt_w_1_2"](self.cbp["mnt_w_1_1"](mc)))
        mnt_h = torch.sigmoid(self.proj["mnt_h_1_2"](self.cbp["mnt_h_1_1"](mc)))
        mnt_s = torch.sigmoid(self.proj["mnt_s_1_2"](self.cbp["mnt_s_1_1"](mc)))
        return {
            "enh_img": enh, "enh_img_imag": enh_imag, "enh_img_real": enh_real,
            "ori_out_1": ori_out, "ori_out_2": ori_out, "seg_out": seg_out,
            "mnt_o_out": mnt_o, "mnt_w_out": mnt_w, "mnt_h_out": mnt_h, "mnt_s_out": mnt_s,
        }


# ---------------- CoarseNet post-processing (numpy, ported from utils) --------
def label2mnt(mnt_s, mnt_w, mnt_h, mnt_o, thresh=0.5):
    """CoarseNet label2mnt: angle is negated vs FingerNet. (N,4) x,y,angle,score."""
    ys, xs = np.where(mnt_s > thresh)
    if len(ys) == 0:
        return np.zeros((0, 4))
    w = np.argmax(mnt_w, axis=-1)
    h = np.argmax(mnt_h, axis=-1)
    o = np.argmax(mnt_o, axis=-1)
    out = np.zeros((len(ys), 4))
    out[:, 0] = xs * 8 + w[ys, xs]
    out[:, 1] = ys * 8 + h[ys, xs]
    ang = (o[ys, xs] * 2 - 89.0) / 180.0 * np.pi
    ang[ang < 0.0] += 2 * np.pi
    out[:, 2] = (-ang) % (2 * np.pi)
    out[:, 3] = mnt_s[ys, xs]
    return out


if __name__ == "__main__":
    import pickle
    WORK = "."  # set to your working dir (reference conversion code)
    ref = pickle.load(open(f"{WORK}/data/crops/coarsenet_ref_maps.pkl", "rb"))
    model = CoarseNetTorch(f"{WORK}/data/crops/coarsenet_keras_weights.pkl").cuda()
    print("=== CoarseNet FCN map verification (torch vs TF1) ===")
    with torch.no_grad():
        for nm, d in ref.items():
            inp = np.transpose(d["input"], (0, 3, 1, 2))         # NHWC -> NCHW
            t = torch.tensor(inp, dtype=torch.float32).cuda()
            out = model(t)
            print(f"\n{nm}  input {tuple(d['input'].shape)}")
            for k, ref_map in d["outs"].items():
                rm = np.transpose(ref_map, (0, 3, 1, 2))         # NHWC -> NCHW
                tm = out[k].cpu().numpy()
                diff = np.abs(rm - tm)
                print(f"  {k:14} shape{str(tuple(tm.shape)):20} "
                      f"maxabs {diff.max():.3e}  mean {diff.mean():.3e}")
