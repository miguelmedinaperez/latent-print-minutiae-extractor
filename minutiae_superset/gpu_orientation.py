"""GPU (torch.fft) port of MinutiaeNet's STFT ridge-flow orientation.

Faithful reimplementation of FastEnhanceTexture + get_maps_STFT (postproc.py) that
produces the same `dir_map` for `fuse_minu_orientation`, but runs the heavy FFTs on
the GPU in batch. The STFT is ~80% of the CPU post-processing, so this accelerates the
refined-angle path without changing the angle (vs ~3.7x faster but network-angle from
`refine_orientation=False`).

Numerics matched to numpy: FFTs in double precision (numpy.fft upcasts float32->
complex128), gradient norms in float32, `'symmetric'` padding, the in-place sequential
histogram smoothing in local_STFT.analysis, and strict-local-max peak selection. The
final block-grid gaussian smoothing reuses the CPU smooth_dir_map (tiny array).
"""
from __future__ import annotations

import math
import numpy as np
import torch

from . import postproc as P


# ---- FastEnhanceTexture ----------------------------------------------------
def _grad_norm(x):
    """np.gradient magnitude (edge_order=1) in float32, +1e-6 — matches compute_gradient_norm."""
    x = x.to(torch.float32)
    gx = torch.zeros_like(x)            # d/d axis0 (rows)
    gx[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2
    gx[0, :] = x[1, :] - x[0, :]
    gx[-1, :] = x[-1, :] - x[-2, :]
    gy = torch.zeros_like(x)            # d/d axis1 (cols)
    gy[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
    gy[:, 0] = x[:, 1] - x[:, 0]
    gy[:, -1] = x[:, -1] - x[:, -2]
    return torch.sqrt(gx * gx + gy * gy) + 1e-6


def _lowpass(x, L, n):
    """LowpassFiltering: zero-pad to nxn, FFT(double), *L (centered), IFFT, crop."""
    h, w = x.shape
    padded = torch.zeros(n, n, dtype=torch.float64, device=x.device)
    padded[:h, :w] = x.to(torch.float64)
    f = torch.fft.fftshift(torch.fft.fft2(padded))
    f = f * L
    rec = torch.fft.ifft2(torch.fft.fftshift(f)).real
    return rec[:h, :w]


def fast_enhance_texture(img, sigma=2.5):
    """img: (H,W) torch tensor. Returns enhanced texture (float64), matching FastEnhanceTexture."""
    h, w = img.shape
    n = int(2 ** math.ceil(math.log(max(h, w), 2)))
    c = torch.arange(-n // 2, n // 2, dtype=torch.float64, device=img.device)
    xx = c.view(1, -1)
    yy = c.view(-1, 1)
    r = (torch.sqrt(xx * xx + yy * yy) + 0.0001) / n
    L = 1.0 / (1 + (2 * math.pi * r * sigma) ** 4)

    img32 = img.to(torch.float32)
    img_low = _lowpass(img32, L, n)
    g1 = _lowpass(_grad_norm(img32), L, n)
    g2 = _lowpass(_grad_norm(img_low), L, n)
    diff = g1 - g2
    ar1 = torch.abs(g1)
    diff = torch.where(ar1 > 1, diff / ar1, torch.zeros_like(diff))
    cmin, cmax = 0.3, 0.7
    weight = (diff - cmin) / (cmax - cmin)
    weight = torch.where(diff < cmin, torch.zeros_like(weight), weight)
    weight = torch.where(diff > cmax, torch.ones_like(weight), weight)
    u = weight * img_low + (1 - weight) * img32.to(torch.float64)
    temp = img32.to(torch.float64) - u
    lim = 20
    v = (temp + lim) * 255 / (2 * lim)
    return torch.clamp(v, 0, 255)


# ---- symmetric pad (numpy 'symmetric', not torch 'reflect') -----------------
def _sym_pad(x, p):
    x = torch.cat([x[:p].flip(0), x, x[-p:].flip(0)], 0)
    x = torch.cat([x[:, :p].flip(1), x, x[:, -p:].flip(1)], 1)
    return x


# ---- per-block STFT direction (cached geometry per device/size) -------------
_CACHE = {}


def _geom(patch_size, device):
    key = (patch_size, str(device))
    if key in _CACHE:
        return _CACHE[key]
    ps = patch_size
    c = torch.arange(-ps // 2, ps // 2, dtype=torch.float64, device=device)
    x = c.view(1, -1).expand(ps, ps)        # cols
    y = c.view(-1, 1).expand(ps, ps)        # rows
    r = torch.sqrt(x * x + y * y) + 0.0001
    RMIN, RMAX = 3, 18
    FLOW, FHIGH = ps / RMAX, ps / RMIN
    dBPass = (1.0 / (1 + (r / FHIGH) ** 4)) * (1.0 / (1 + (FLOW / r) ** 4))
    nd = 16
    drc = torch.atan2(y, x)
    drc = torch.where(drc < 0, drc + math.pi, drc)
    dir_ind = torch.floor(drc / (math.pi / nd)).to(torch.long)
    dir_ind = torch.where(dir_ind == nd, torch.zeros_like(dir_ind), dir_ind)
    binmat = torch.zeros(ps * ps, nd, dtype=torch.float64, device=device)
    binmat[torch.arange(ps * ps, device=device), dir_ind.reshape(-1)] = 1.0
    sig = ps / 3
    weight = torch.exp(-(x * x + y * y) / (sig * sig))
    out = (weight, dBPass, binmat)
    _CACHE[key] = out
    return out


@torch.no_grad()
def get_maps_STFT_gpu(img2d, device, patch_size=64, block_size=16, preprocess=True):
    """GPU dir_map matching get_maps_STFT(img, 64, 16, preprocess)."""
    t = torch.as_tensor(np.asarray(img2d), dtype=torch.float64, device=device)
    if preprocess:
        t = fast_enhance_texture(t, sigma=2.5)
    ovp = (patch_size - block_size) // 2
    tp = _sym_pad(t, ovp)
    h, w = tp.shape
    blkH = (h - patch_size) // block_size + 1
    blkW = (w - patch_size) // block_size + 1

    weight, dBPass, binmat = _geom(patch_size, device)

    patches = tp.unfold(0, patch_size, block_size).unfold(1, patch_size, block_size)
    patches = patches.reshape(blkH * blkW, patch_size, patch_size).contiguous()   # (B,ps,ps)

    p = patches * weight
    p = p - p.mean(dim=(1, 2), keepdim=True)
    nrm = torch.sqrt((p * p).sum(dim=(1, 2), keepdim=True))
    p = p / (nrm + 1e-6)
    fsh = torch.fft.fftshift(torch.fft.fft2(p), dim=(-2, -1))
    fsh = fsh * dBPass
    energy = torch.abs(fsh)
    energy = energy / (energy.sum(dim=(1, 2), keepdim=True) + 1e-5)

    B = energy.shape[0]
    dir_norm = energy.reshape(B, -1) @ binmat                # (B,16)
    nd = 16
    pad = torch.empty(B, nd + 2, dtype=torch.float64, device=device)
    pad[:, 1:nd + 1] = dir_norm
    pad[:, 0] = dir_norm[:, nd - 1]
    pad[:, nd + 1] = dir_norm[:, 0]
    # in-place sequential smoothing: sm[i] uses sm[i-1] (updated) + orig[i]*4 + orig[i+1]
    orig = pad.clone()
    sm = pad
    for i in range(1, nd + 1):
        sm[:, i] = (sm[:, i - 1] + orig[:, i] * 4 + orig[:, i + 1]) / 6
    sm[:, 0] = sm[:, nd]
    sm[:, nd + 1] = sm[:, 1]

    center = sm[:, 1:nd + 1]
    left = sm[:, 0:nd]
    right = sm[:, 2:nd + 2]
    is_peak = (center > left) & (center > right)             # strict local maxima
    conf = torch.where(is_peak, center, torch.full_like(center, -1e30))
    peak_bin = conf.argmax(dim=1)                            # 0-based (i = peak_bin+1)
    has_peak = is_peak.any(dim=1)
    ori_interval = math.pi / nd
    ori = peak_bin.to(torch.float64) * ori_interval + ori_interval / 2 + math.pi / 2
    dir_flat = torch.where(has_peak, ori, torch.full_like(ori, -10.0))

    dir_map = dir_flat.reshape(blkH, blkW).cpu().numpy()
    return P.smooth_dir_map(dir_map)
