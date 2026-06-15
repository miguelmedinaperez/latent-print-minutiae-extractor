"""FineNet patch classifier (InceptionResNetV2 + 2-class head), PyTorch.

Port of FineNet_model.py from MinutiaeNet (github.com/luannd/MinutiaeNet, MIT). Input
is a 224x224x3 uint8 [0,255] patch; output logits -> softmax[:,0] = P(minutia).
Built empty (deploy) or from a Keras weight-dump (conversion); module structure and
thus state_dict keys are identical either way.
"""
from __future__ import annotations

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loader:
    """Keras weights by name; cursor for unnamed stem/branch/mixed convs.
    Empty (no pkl) -> default init, weights supplied later via load_state_dict."""

    def __init__(self, keras_pkl: str | None = None):
        if keras_pkl is None:
            self.d: dict = {}
        else:
            with open(keras_pkl, "rb") as f:
                self.d = {l["name"]: l["weights"] for l in pickle.load(f)}
        self.k = 0
        self.used = set()

    def get(self, name=None):
        if name is None:
            self.k += 1
            cname, bname = f"conv2d_{self.k}", f"batch_normalization_{self.k}"
        else:
            cname, bname = name, name + "_bn"
        self.used.add(cname)
        self.used.add(bname)
        return self.d.get(cname), self.d.get(bname)


class ConvBN(nn.Module):
    """conv2d_bn mirror. use_bias=True -> conv has bias and NO BatchNorm (Keras rule)."""

    def __init__(self, kw, in_ch, out_ch, k, stride=1, padding=0, use_bias=False,
                 act=True, name=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=use_bias)
        self.act = act
        self.has_bn = not use_bias
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_ch, eps=1e-3)
        cw, bnw = kw.get(name)
        if cw is not None:
            kernel = cw[0]
            assert kernel.shape[2] == in_ch and kernel.shape[3] == out_ch, \
                f"{name}: shape {kernel.shape} vs in{in_ch} out{out_ch}"
            self.conv.weight.data = torch.tensor(np.transpose(kernel, (3, 2, 0, 1)).copy())
            if use_bias:
                self.conv.bias.data = torch.tensor(cw[1].copy())
            if self.has_bn:
                beta, mean, var = bnw
                self.bn.weight.data = torch.ones(out_ch)
                self.bn.bias.data = torch.tensor(beta.copy())
                self.bn.running_mean.data = torch.tensor(mean.copy())
                self.bn.running_var.data = torch.tensor(var.copy())

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.act:
            x = F.relu(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, kw, ch, block_type, block_idx, scale, last_act=True):
        super().__init__()
        self.scale = scale
        self.last_act = last_act
        if block_type == "block35":
            self.b0 = nn.ModuleList([ConvBN(kw, ch, 32, 1)])
            self.b1 = nn.ModuleList([ConvBN(kw, ch, 32, 1), ConvBN(kw, 32, 32, 3, padding=1)])
            self.b2 = nn.ModuleList([ConvBN(kw, ch, 32, 1), ConvBN(kw, 32, 48, 3, padding=1),
                                     ConvBN(kw, 48, 64, 3, padding=1)])
            mixed_ch = 32 + 32 + 64
        elif block_type == "block17":
            self.b0 = nn.ModuleList([ConvBN(kw, ch, 192, 1)])
            self.b1 = nn.ModuleList([ConvBN(kw, ch, 128, 1),
                                     ConvBN(kw, 128, 160, (1, 7), padding=(0, 3)),
                                     ConvBN(kw, 160, 192, (7, 1), padding=(3, 0))])
            self.b2 = None
            mixed_ch = 192 + 192
        else:  # block8
            self.b0 = nn.ModuleList([ConvBN(kw, ch, 192, 1)])
            self.b1 = nn.ModuleList([ConvBN(kw, ch, 192, 1),
                                     ConvBN(kw, 192, 224, (1, 3), padding=(0, 1)),
                                     ConvBN(kw, 224, 256, (3, 1), padding=(1, 0))])
            self.b2 = None
            mixed_ch = 192 + 256
        self.up = ConvBN(kw, mixed_ch, ch, 1, use_bias=True, act=False,
                         name=f"{block_type}_{block_idx}_conv")

    @staticmethod
    def _run(mods, x):
        for m in mods:
            x = m(x)
        return x

    def forward(self, x):
        outs = [self._run(self.b0, x), self._run(self.b1, x)]
        if self.b2 is not None:
            outs.append(self._run(self.b2, x))
        up = self.up(torch.cat(outs, dim=1))
        x = x + up * self.scale
        return F.relu(x) if self.last_act else x


class FineNet(nn.Module):
    def __init__(self, kw: Loader | None = None):
        super().__init__()
        kw = kw or Loader()
        self.stem = nn.ModuleList([
            ConvBN(kw, 3, 32, 3, stride=2),
            ConvBN(kw, 32, 32, 3),
            ConvBN(kw, 32, 64, 3, padding=1),
            ConvBN(kw, 64, 80, 1),
            ConvBN(kw, 80, 192, 3),
        ])
        self.m5_b0 = ConvBN(kw, 192, 96, 1)
        self.m5_b1 = nn.ModuleList([ConvBN(kw, 192, 48, 1), ConvBN(kw, 48, 64, 5, padding=2)])
        self.m5_b2 = nn.ModuleList([ConvBN(kw, 192, 64, 1), ConvBN(kw, 64, 96, 3, padding=1),
                                    ConvBN(kw, 96, 96, 3, padding=1)])
        self.m5_bp = ConvBN(kw, 192, 64, 1)
        self.block35 = nn.ModuleList([ResnetBlock(kw, 320, "block35", i, 0.17) for i in range(1, 11)])
        self.m6_b0 = ConvBN(kw, 320, 384, 3, stride=2)
        self.m6_b1 = nn.ModuleList([ConvBN(kw, 320, 256, 1), ConvBN(kw, 256, 256, 3, padding=1),
                                    ConvBN(kw, 256, 384, 3, stride=2)])
        self.block17 = nn.ModuleList([ResnetBlock(kw, 1088, "block17", i, 0.1) for i in range(1, 21)])
        self.m7_b0 = nn.ModuleList([ConvBN(kw, 1088, 256, 1), ConvBN(kw, 256, 384, 3, stride=2)])
        self.m7_b1 = nn.ModuleList([ConvBN(kw, 1088, 256, 1), ConvBN(kw, 256, 288, 3, stride=2)])
        self.m7_b2 = nn.ModuleList([ConvBN(kw, 1088, 256, 1), ConvBN(kw, 256, 288, 3, padding=1),
                                    ConvBN(kw, 288, 320, 3, stride=2)])
        self.block8 = nn.ModuleList(
            [ResnetBlock(kw, 2080, "block8", i, 0.2) for i in range(1, 10)] +
            [ResnetBlock(kw, 2080, "block8", 10, 1.0, last_act=False)])
        self.conv_7b = ConvBN(kw, 2080, 1536, 1, name="conv_7b")
        self.fc = nn.Linear(1536, 2)
        if "predictions" in kw.d:
            dense = kw.d["predictions"]
            kw.used.add("predictions")
            self.fc.weight.data = torch.tensor(dense[0].T.copy())
            self.fc.bias.data = torch.tensor(dense[1].copy())
        self.eval()

    @staticmethod
    def _seq(mods, x):
        for m in mods:
            x = m(x)
        return x

    def forward(self, x):
        for i, m in enumerate(self.stem):
            x = m(x)
            if i == 2:
                x = F.max_pool2d(x, 3, stride=2)
            if i == 4:
                x = F.max_pool2d(x, 3, stride=2)
        bp = self.m5_bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        x = torch.cat([self.m5_b0(x), self._seq(self.m5_b1, x), self._seq(self.m5_b2, x), bp], 1)
        for b in self.block35:
            x = b(x)
        x = torch.cat([self.m6_b0(x), self._seq(self.m6_b1, x), F.max_pool2d(x, 3, stride=2)], 1)
        for b in self.block17:
            x = b(x)
        x = torch.cat([self._seq(self.m7_b0, x), self._seq(self.m7_b1, x),
                       self._seq(self.m7_b2, x), F.max_pool2d(x, 3, stride=2)], 1)
        for b in self.block8:
            x = b(x)
        x = self.conv_7b(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
