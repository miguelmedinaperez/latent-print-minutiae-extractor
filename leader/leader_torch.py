"""PyTorch port of PyFing/LEADER (Keras) so it can be fine-tuned on the GPU.

Architecture mirrors pyfing.minutiae.Leader._build_model exactly. Each parametrized layer is
auto-constructed from the dumped Keras weight shapes (data/crops/leader_weights.npz +
leader_layers.json from scripts/_dump_leader.py); only the forward topology is hand-written.

Heads: pos (1ch sigmoid detection map), dir (2ch cos/sin -> atan2 angle), typ (1ch sigmoid).
forward() returns the raw head tensors; extract() adds LEADER's NMS (gaussian-blur + 7x7 maxpool
local-max) to recover minutiae, matching the Keras pipeline. Verified to ~1e-5 vs Keras (main()).

Run in zen_meninsky (torch + GPU). Inputs are raw 0-255 (LayerNorm handles scale).
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GELU = lambda x: F.gelu(x, approximate="none")


class ChannelLN(nn.Module):
    """Keras LayerNormalization(axis=-1) on NCHW: normalize over C per (n,h,w)."""
    def __init__(self, c, eps=1e-3):
        super().__init__()
        self.g = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c)); self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True); var = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.g[None, :, None, None] + self.b[None, :, None, None]


class SepConv(nn.Module):
    """Keras SeparableConv2D: depthwise (no bias) then pointwise 1x1 (bias)."""
    def __init__(self, ci, co, k, pad):
        super().__init__()
        self.dw = nn.Conv2d(ci, ci, k, padding=pad, groups=ci, bias=False)
        self.pw = nn.Conv2d(ci, co, 1, bias=True)

    def forward(self, x):
        return self.pw(self.dw(x))


class LeaderTorch(nn.Module):
    def __init__(self, weights_npz, layers_json):
        super().__init__()
        W = np.load(weights_npz)
        meta = {m["name"]: m for m in json.load(open(layers_json))}
        self.order = [m["name"] for m in json.load(open(layers_json))]
        self.cfg = meta
        self.L = nn.ModuleDict()
        for name, m in meta.items():
            t = m["type"]
            if t == "Conv2D":
                k = W[f"{name}::0"]; kh, kw, ci, co = k.shape
                d = int(m["config"]["dilation_rate"][0])
                self.L[name] = nn.Conv2d(ci, co, (kh, kw), padding=d * (kh // 2), dilation=d,
                                         bias=(f"{name}::1" in W.files))
            elif t == "SeparableConv2D":
                dwk = W[f"{name}::0"]; kh, kw, ci, _ = dwk.shape
                co = W[f"{name}::1"].shape[3]
                self.L[name] = SepConv(ci, co, (kh, kw), kh // 2)
            elif t == "DepthwiseConv2D":
                dwk = W[f"{name}::0"]; kh, kw, ci, _ = dwk.shape
                self.L[name] = nn.Conv2d(ci, ci, (kh, kw), padding=kh // 2, groups=ci, bias=False)
            elif t == "LayerNormalization":
                self.L[name] = ChannelLN(W[f"{name}::0"].shape[0], eps=float(m["config"]["epsilon"]))
        self._load(W)
        # fixed gaussian blur kernel for the inference NMS (cv.getGaussianKernel(5,0))
        import cv2 as cv
        gw = cv.getGaussianKernel(5, 0).astype(np.float32); gw = np.outer(gw, gw)
        self.register_buffer("gblur", torch.tensor(gw)[None, None])

    def _load(self, W):
        for name, m in self.cfg.items():
            t = m["type"]; mod = self.L[name] if name in self.L else None
            if t == "Conv2D":
                mod.weight.data = torch.tensor(np.transpose(W[f"{name}::0"], (3, 2, 0, 1)).copy())
                if f"{name}::1" in W.files:
                    mod.bias.data = torch.tensor(W[f"{name}::1"].copy())
            elif t == "SeparableConv2D":
                mod.dw.weight.data = torch.tensor(np.transpose(W[f"{name}::0"], (2, 3, 0, 1)).copy())
                mod.pw.weight.data = torch.tensor(np.transpose(W[f"{name}::1"], (3, 2, 0, 1)).copy())
                mod.pw.bias.data = torch.tensor(W[f"{name}::2"].copy())
            elif t == "DepthwiseConv2D":
                mod.weight.data = torch.tensor(np.transpose(W[f"{name}::0"], (2, 3, 0, 1)).copy())
            elif t == "LayerNormalization":
                mod.g.data = torch.tensor(W[f"{name}::0"].copy()); mod.b.data = torch.tensor(W[f"{name}::1"].copy())

    # ---- blocks (mirror Leader static methods) ----
    def _stem(self, name, x, mp):
        x1 = self.L[f"{name}_conv"](x); x2 = self.L[f"{name}_dilated_conv"](x)
        x = torch.cat([x1, x2], 1); x = GELU(self.L[f"{name}_ln"](x))
        return F.max_pool2d(x, 2, ceil_mode=True) if mp else F.avg_pool2d(x, 2, ceil_mode=True)

    def _sep(self, name, x):
        return GELU(self.L[f"{name}_ln"](self.L[f"{name}_conv"](x)))

    def _ibc(self, name, x):
        xin = x
        x = self.L[f"{name}_ln"](self.L[f"{name}_depthwise_conv"](x))
        x = GELU(self.L[f"{name}_conv_exp"](x))
        x = self.L[f"{name}_conv_red"](x)
        x = xin + x
        return self.L[f"{name}_conv_adj"](x)

    def _down(self, x):
        c1 = x.shape[1] // 2
        a = F.max_pool2d(x[:, :c1], 2, ceil_mode=True); b = F.avg_pool2d(x[:, c1:], 2, ceil_mode=True)
        return torch.cat([a, b], 1)

    def _up(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x if skip is None else torch.cat([x, skip], 1)

    def _head(self, name, x):
        x = self._ibc(f"{name}_ibc", x)
        x = self._up(x)
        x = self.L[f"{name}_ln"](self.L[f"{name}_conv1"](x))
        return GELU(self.L[f"{name}_conv2"](x))

    def forward(self, x):
        s = self._stem("stem0", x, True); x1 = self._stem("stem1", x, False)
        x = s
        skips = []
        for i, fc in enumerate([16, 32, 64, 128]):
            x = self._sep(f"enc0_{i}", x); skips.append(x); x = self._down(x)
        for i in reversed(range(4)):
            x = self._sep(f"dec0_{i}", x); x = self._up(x, skips[i])
        d = [GELU(self.L[f"attention_conv_d{r}"](x)) for r in (1, 3, 6)]
        xf = torch.sigmoid(self.L["attention_s"](torch.cat(d, 1)))
        x = x * xf
        x = torch.cat([x, x1], 1)
        skips = []
        for i in range(4):
            x = self._ibc(f"enc1_{i}", x); skips.append(x); x = self._down(x)
        for i in reversed(range(4)):
            x = self._ibc(f"dec1_{i}", x); x = self._up(x, skips[i])
        h = [self._head(f"head_{k}", x) for k in range(3)]
        pos = torch.sigmoid(self.L["head_conv_pos"](h[0]))
        dir2 = self.L["head_conv_dir"](h[1])
        typ = torch.sigmoid(self.L["head_conv_typ"](h[2]))
        return pos, dir2, typ

    @torch.no_grad()
    def extract(self, pos, dir2, typ, q=0.01):
        """LEADER NMS -> list of (x,y,angle,type,quality) per the Keras _get_minutiae."""
        gb = F.conv2d(pos, self.gblur, padding=2)
        mx = F.max_pool2d(gb, 7, stride=1, padding=3)
        nms = gb * (gb == mx).float()
        ang = torch.atan2(dir2[:, 1:2], dir2[:, 0:1])
        out = []
        for b in range(pos.shape[0]):
            ys, xs = torch.where(nms[b, 0] >= q)
            out.append([(int(x), int(y), float(ang[b, 0, y, x]),
                         "E" if typ[b, 0, y, x] >= 0.5 else "B", float(nms[b, 0, y, x]))
                        for y, x in zip(ys.tolist(), xs.tolist())])
        return out


def main():
    """Smoke test: load the port and run a forward pass. If a Keras parity dump
    (leader_parity.npz, produced by dump_leader.py) is present, also report max|Δ| vs Keras."""
    from pathlib import Path
    import os
    W = Path(__file__).resolve().parent / "weights"
    m = LeaderTorch(str(W / "leader_weights.npz"), str(W / "leader_layers.json")).eval()
    parity = W / "leader_parity.npz"
    if parity.exists():
        P = np.load(str(parity))
        x = torch.tensor(P["x"]).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            pos, dir2, typ = m(x)
        for nm, t, k in [("pos", pos, P["pos"]), ("dir2", dir2, P["dir2"]), ("typ", typ, P["typ"])]:
            k = k.transpose(0, 3, 1, 2)
            print(f"{nm:5}: max|Δ| {np.abs(t.numpy()-k).max():.2e} vs Keras (range {k.min():.3f}..{k.max():.3f})")
    else:
        x = torch.zeros(1, 1, 320, 320)
        with torch.no_grad():
            pos, dir2, typ = m(x)
        print(f"loaded OK. forward(1,1,320,320) -> pos{tuple(pos.shape)} dir{tuple(dir2.shape)} typ{tuple(typ.shape)}; "
              f"pos range {pos.min():.3f}..{pos.max():.3f}")
        print("(drop leader_parity.npz from dump_leader.py into ./weights for the full Keras parity check)")


if __name__ == "__main__":
    main()
