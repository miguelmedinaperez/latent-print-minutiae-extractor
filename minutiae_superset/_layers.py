"""Shared layers for the FingerNet / CoarseNet ports.

A `WSrc` decouples architecture construction from weight values: built from a Keras
weight-dump (conversion time) it fills the layers; built empty (deploy time) it leaves
default init so weights are supplied later via ``load_state_dict``. Either way the
module structure — and therefore the state_dict keys — is identical.
"""
from __future__ import annotations

import pickle
import numpy as np
import torch
import torch.nn as nn


class WSrc:
    """Weight source for FingerNet/CoarseNet, keyed by Keras layer name."""

    def __init__(self, keras_pkl: str | None = None):
        if keras_pkl is None:
            self.d: dict = {}
        else:
            with open(keras_pkl, "rb") as f:
                self.d = {l["name"]: l["weights"] for l in pickle.load(f)}

    def conv(self, name, in_ch, out_ch, k, dilation=1, bias=True):
        pad = dilation * (k - 1) // 2
        m = nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation, bias=bias)
        if name in self.d:
            w = self.d[name]
            m.weight.data = torch.tensor(np.transpose(w[0], (3, 2, 0, 1)).copy())
            if bias:
                m.bias.data = torch.tensor(w[1].copy())
        return m

    def bn(self, name, ch):
        m = nn.BatchNorm2d(ch, eps=1e-3)
        if name in self.d:
            gamma, beta, mean, var = self.d[name]
            m.weight.data = torch.tensor(gamma.copy())
            m.bias.data = torch.tensor(beta.copy())
            m.running_mean.data = torch.tensor(mean.copy())
            m.running_var.data = torch.tensor(var.copy())
        return m

    def prelu(self, name, ch):
        m = nn.PReLU(ch)
        if name in self.d:
            m.weight.data = torch.tensor(self.d[name][0].reshape(-1).copy())
        return m


class CBP(nn.Module):
    """conv_bn_prelu: Conv -> BN(eps=1e-3) -> PReLU(per-channel)."""

    def __init__(self, src, conv_name, bn_name, prelu_name, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.c = src.conv(conv_name, in_ch, out_ch, k, dilation)
        self.b = src.bn(bn_name, out_ch)
        self.p = src.prelu(prelu_name, out_ch)

    def forward(self, x):
        return self.p(self.b(self.c(x)))


def gausslabel_matrix(length=180, stride=2, sigma=3):
    """Circular gaussian orientation-smoothing kernel (90x90), matches gausslabel()."""
    arr = np.arange(stride // 2, length, stride)
    gp = np.exp(-0.5 * ((np.arange(length + 1) - length / 2) / sigma) ** 2)
    delta = np.abs(arr[:, None] - arr[None, :])
    delta = np.minimum(delta, length - delta).astype(int) + length // 2
    return gp[delta].astype(np.float32)


class FineGabor(nn.Module):
    """Stored 25x25 Gabor filter bank conv (1 -> 90)."""

    def __init__(self, src, name):
        super().__init__()
        self.c = src.conv(name, 1, 90, 25, bias=True)

    def forward(self, x):
        return self.c(x)
