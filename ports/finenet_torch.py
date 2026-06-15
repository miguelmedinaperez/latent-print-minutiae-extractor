"""PyTorch port of FineNet (InceptionResNetV2 + 2-class head) with Keras weights.

The architecture mirrors external/MinutiaeNet/FineNet/FineNet_model.py EXACTLY
(same branch order in every block), so weights dumped from Keras in model.layers
order (finenet_dump_weights.py) can be loaded in-order via a cursor with shape
asserts. Keras specifics reproduced:
  * conv2d_bn: Conv(use_bias=False) -> BN(scale=False, eps=1e-3) -> relu.
    BN with scale=False has NO gamma (fixed 1); weights are [beta, mean, var].
  * block `up` conv: use_bias=True, activation=None -> conv WITH bias, no BN, no relu.
  * inputs are uint8 [0,255] 3-channel (NO [-1,1] normalization in FineNet inference).

`build_finenet(weights_pkl)` returns the loaded model. `verify()` re-runs the 44px
zero-shot on our val crops and must match the TF1 result (~72%) to trust the port.
"""
from __future__ import annotations

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loader:
    """Loads Keras weights BY NAME (model.layers order is topological, not creation
    order — but names encode creation order, and a conv2d_bn's conv+BN share an index)."""
    def __init__(self, path):
        with open(path, "rb") as f:
            L = pickle.load(f)
        self.d = {l["name"]: l["weights"] for l in L}
        self.k = 0          # auto counter for unnamed branch/stem/mixed convs
        self.used = set()

    def get(self, name=None):
        if name is None:
            self.k += 1
            cname, bname = f"conv2d_{self.k}", f"batch_normalization_{self.k}"
        else:
            cname, bname = name, name + "_bn"
        self.used.add(cname); self.used.add(bname)
        return self.d[cname], self.d.get(bname)


class ConvBN(nn.Module):
    """conv2d_bn mirror. use_bias=True -> conv has bias and NO BatchNorm (Keras rule)."""
    def __init__(self, kw, in_ch, out_ch, k, stride=1, padding=0, use_bias=False,
                 act=True, name=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=use_bias)
        self.act = act
        cw, bnw = kw.get(name)
        self.has_bn = bnw is not None
        assert self.has_bn == (not use_bias), f"{name}: bn/use_bias mismatch"
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_ch, eps=1e-3)

        kernel = cw[0]  # (H, W, in, out)
        assert kernel.shape[2] == in_ch and kernel.shape[3] == out_ch, \
            f"{name or 'conv2d_'+str(kw.k)}: shape {kernel.shape} vs in{in_ch} out{out_ch}"
        self.conv.weight.data = torch.tensor(np.transpose(kernel, (3, 2, 0, 1)).copy())
        if use_bias:
            self.conv.bias.data = torch.tensor(cw[1].copy())
        if self.has_bn:
            beta, mean, var = bnw
            self.bn.weight.data = torch.ones(out_ch)        # gamma fixed (scale=False)
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
    """inception_resnet_block mirror (block35 / block17 / block8)."""
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

    def _run(self, mods, x):
        for m in mods:
            x = m(x)
        return x

    def forward(self, x):
        outs = [self._run(self.b0, x), self._run(self.b1, x)]
        if self.b2 is not None:
            outs.append(self._run(self.b2, x))
        mixed = torch.cat(outs, dim=1)
        up = self.up(mixed)
        x = x + up * self.scale
        if self.last_act:
            x = F.relu(x)
        return x


class FineNetTorch(nn.Module):
    def __init__(self, kw):
        super().__init__()
        # Stem
        self.stem = nn.ModuleList([
            ConvBN(kw, 3, 32, 3, stride=2),            # conv2d_1 (valid)
            ConvBN(kw, 32, 32, 3),                     # conv2d_2 (valid)
            ConvBN(kw, 32, 64, 3, padding=1),          # conv2d_3 (same)
            ConvBN(kw, 64, 80, 1),                     # conv2d_4 (after maxpool)
            ConvBN(kw, 80, 192, 3),                    # conv2d_5 (valid)
        ])
        # mixed_5b
        self.m5_b0 = ConvBN(kw, 192, 96, 1)
        self.m5_b1 = nn.ModuleList([ConvBN(kw, 192, 48, 1), ConvBN(kw, 48, 64, 5, padding=2)])
        self.m5_b2 = nn.ModuleList([ConvBN(kw, 192, 64, 1), ConvBN(kw, 64, 96, 3, padding=1),
                                    ConvBN(kw, 96, 96, 3, padding=1)])
        self.m5_bp = ConvBN(kw, 192, 64, 1)            # after avgpool
        self.block35 = nn.ModuleList([ResnetBlock(kw, 320, "block35", i, 0.17) for i in range(1, 11)])
        # mixed_6a
        self.m6_b0 = ConvBN(kw, 320, 384, 3, stride=2)
        self.m6_b1 = nn.ModuleList([ConvBN(kw, 320, 256, 1), ConvBN(kw, 256, 256, 3, padding=1),
                                    ConvBN(kw, 256, 384, 3, stride=2)])
        self.block17 = nn.ModuleList([ResnetBlock(kw, 1088, "block17", i, 0.1) for i in range(1, 21)])
        # mixed_7a
        self.m7_b0 = nn.ModuleList([ConvBN(kw, 1088, 256, 1), ConvBN(kw, 256, 384, 3, stride=2)])
        self.m7_b1 = nn.ModuleList([ConvBN(kw, 1088, 256, 1), ConvBN(kw, 256, 288, 3, stride=2)])
        self.m7_b2 = nn.ModuleList([ConvBN(kw, 1088, 256, 1), ConvBN(kw, 256, 288, 3, padding=1),
                                    ConvBN(kw, 288, 320, 3, stride=2)])
        self.block8 = nn.ModuleList(
            [ResnetBlock(kw, 2080, "block8", i, 0.2) for i in range(1, 10)] +
            [ResnetBlock(kw, 2080, "block8", 10, 1.0, last_act=False)])
        self.conv_7b = ConvBN(kw, 2080, 1536, 1, name="conv_7b")
        # head
        dense = kw.d["predictions"]  # [kernel(1536,2), bias(2)]
        kw.used.add("predictions")
        self.fc = nn.Linear(1536, 2)
        self.fc.weight.data = torch.tensor(dense[0].T.copy())
        self.fc.bias.data = torch.tensor(dense[1].copy())

    def forward(self, x):
        for i, m in enumerate(self.stem):
            x = m(x)
            if i == 2:  # maxpool after conv2d_3
                x = F.max_pool2d(x, 3, stride=2)
            if i == 4:  # maxpool after conv2d_5
                x = F.max_pool2d(x, 3, stride=2)
        # mixed_5b
        bp = self.m5_bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        x = torch.cat([self.m5_b0(x), self._seq(self.m5_b1, x), self._seq(self.m5_b2, x), bp], 1)
        for b in self.block35:
            x = b(x)
        # mixed_6a
        x = torch.cat([self.m6_b0(x), self._seq(self.m6_b1, x), F.max_pool2d(x, 3, stride=2)], 1)
        for b in self.block17:
            x = b(x)
        # mixed_7a
        x = torch.cat([self._seq(self.m7_b0, x), self._seq(self.m7_b1, x),
                       self._seq(self.m7_b2, x), F.max_pool2d(x, 3, stride=2)], 1)
        for b in self.block8:
            x = b(x)
        x = self.conv_7b(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

    @staticmethod
    def _seq(mods, x):
        for m in mods:
            x = m(x)
        return x


def build_finenet(weights_pkl):
    kw = Loader(weights_pkl)
    model = FineNetTorch(kw)
    unused = set(kw.d) - kw.used
    assert not unused, f"unused Keras layers: {sorted(unused)[:6]}..."
    model.eval()
    return model


if __name__ == "__main__":
    import json, os, random, cv2
    WORK = "."  # set to your working dir (reference conversion code)
    model = build_finenet(f"{WORK}/data/crops/finenet_keras_weights.pkl").cuda()
    print("loaded; all Keras layers consumed in order.")

    val = [json.loads(l) for l in open(f"{WORK}/data/dataset/val.jsonl")]
    pos = [r for r in val if r["label"] == "positive"]
    neg = [r for r in val if r["label"] == "negative"]
    random.seed(0); random.shuffle(pos); random.shuffle(neg)
    sample = pos[:60] + neg[:60]

    def patch44(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        c = img.shape[0] // 2
        p = img[c - 22:c + 22, c - 22:c + 22]
        p = cv2.resize(p, (224, 224), interpolation=cv2.INTER_NEAREST)
        return np.stack([p, p, p], 0).astype(np.float32)  # (3,224,224) [0,255]

    X = torch.tensor(np.stack([patch44(f"{WORK}/data/crops/{r['image']}") for r in sample])).cuda()
    with torch.no_grad():
        pred = model(X).argmax(1).cpu().numpy()  # 0 = minutia
    per = {"positive": [0, 0], "negative": [0, 0]}
    for r, p in zip(sample, pred):
        per[r["label"]][1] += 1
        per[r["label"]][0] += int((p == 0) == (r["label"] == "positive"))
    pa, na = per["positive"], per["negative"]
    bal = (pa[0] / pa[1] + na[0] / na[1]) / 2
    print(f"torch FineNet zero-shot 44px | positives {pa[0]}/{pa[1]}  negatives {na[0]}/{na[1]}  "
          f"| balanced {bal:.1%}   (TF1 was 71.7%)")
