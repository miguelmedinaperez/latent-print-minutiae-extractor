"""Universal LEADER minutiae extractor — Python API.

    from leader import MinutiaeExtractor
    ex = MinutiaeExtractor()                       # loads the fine-tuned universal model
    minutiae = ex.extract(gray_image, dpi=500)     # -> [{'x','y','angle','quality','type'}, ...]
    batch    = ex.extract_batch([img1, img2])      # -> [[...], [...]]   (padded to a common size)

Works on latent FINGERPRINTS and PALMPRINTS at 500 dpi (other DPIs are resampled). CPU or GPU
(auto). Angle is in radians, LEADER's convention (negate if your GT uses the opposite sign).

`type` is "E" (ridge ending) or "B" (bifurcation), from LEADER's type head. NOTE: the fine-tuning
supervised detection (`pos`) and direction (`dir`); the type head is inherited from LEADER's
pretraining and was not re-validated on latents — treat `type` as best-effort.
"""
from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from .leader_torch import LeaderTorch

_W = Path(__file__).resolve().parent / "weights"


def _pad32(img, value):
    h, w = img.shape
    H, Wd = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    t, l = (H - h) // 2, (Wd - w) // 2
    return cv.copyMakeBorder(img, t, H - h - t, l, Wd - w - l, cv.BORDER_CONSTANT, value=value), t, l


class MinutiaeExtractor:
    def __init__(self, weights_dir=_W, finetuned="leader_universal_deploy.pt", device=None,
                 half=None, compile=False, tta=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 autocast on GPU: ~25% faster, minutiae preserved (positions + angles). Off on CPU.
        self.half = (self.device == "cuda") if half is None else half
        # test-time augmentation: average the detection map over 4 flips. ~+0.02 loc AP (esp. palms)
        # at ~5x forward cost. Opt-in — default off keeps the fast single-pass path.
        self.tta = tta
        self.model = LeaderTorch(str(Path(weights_dir) / "leader_weights.npz"),
                                 str(Path(weights_dir) / "leader_layers.json")).to(self.device).eval()
        ft = Path(weights_dir) / finetuned
        if finetuned and ft.exists():
            self.model.load_state_dict(torch.load(str(ft), map_location=self.device))
        # opt-in CUDA-graph compile: ~2.5x faster, identical minutiae, BUT recompiles per input size
        # (best for fixed-size / high-volume same-size workloads; ~1 min warmup on first call).
        self.compiled = bool(compile and self.device == "cuda")   # GPU-only; introspectable
        self._fwd = (torch.compile(self.model, mode="reduce-overhead")
                     if self.compiled else self.model)

    def _forward(self, x):
        if self.half and self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                return self._fwd(x)
        return self._fwd(x)

    def _infer(self, x):
        """Forward, optionally with test-time augmentation: average the pos detection map over the 4
        flips (un-flipped back); dir/typ from the identity pass. Improves detection AP, esp. on palms.

        With ``compile=True`` (CUDA graphs / ``reduce-overhead``) the network's outputs are *views into
        a static replay buffer* that the NEXT forward overwrites. TTA runs the model 5 times while
        keeping the accumulator alive across calls, so each detection map must be cloned out of that
        buffer before the following forward clobbers it — otherwise torch raises
        "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run".
        The single-pass path is safe: its outputs are consumed before any re-invocation."""
        if not self.tta:
            return self._forward(x)
        acc = None
        for fx in (False, True):
            for fy in (False, True):
                xx = x
                if fx: xx = torch.flip(xx, dims=[3])
                if fy: xx = torch.flip(xx, dims=[2])
                pos = self._forward(xx)[0].clone()      # own the data before the next forward reuses the buffer
                if fx: pos = torch.flip(pos, dims=[3])
                if fy: pos = torch.flip(pos, dims=[2])
                acc = pos if acc is None else acc + pos
        _, dir2, typ = self._forward(x)
        return acc / 4.0, dir2, typ

    @staticmethod
    def _prep(img, dpi):
        img = np.asarray(img)
        if img.ndim == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if dpi != 500:
            s = 500 / dpi
            img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_CUBIC)
        return img

    def _decode(self, mns, ox, oy, w, h, scale):
        out = []
        for mx, my, ang, ty, ql in mns:
            x, y = (mx - ox), (my - oy)
            if 0 <= x < w and 0 <= y < h:
                out.append({"x": int(round(x * scale)), "y": int(round(y * scale)),
                            "angle": float(ang), "quality": float(ql), "type": ty})
        return out

    @torch.no_grad()
    def extract(self, image, dpi=500, quality=0.1):
        """Extract minutiae from one grayscale image. Returns a list of dicts."""
        img = self._prep(image, dpi); h, w = img.shape
        scale = dpi / 500 if dpi != 500 else 1.0
        pad, t, l = _pad32(img.astype(np.float32), float(img[0, 0]))
        x = torch.tensor(pad[None, None], dtype=torch.float32, device=self.device)
        pos, dir2, typ = self._infer(x)
        return self._decode(self.model.extract(pos, dir2, typ, q=quality)[0], l, t, w, h, scale)

    @torch.no_grad()
    def extract_batch(self, images, dpi=500, quality=0.1):
        """Extract from many images in one forward pass (padded to a common size). Best throughput
        when images are similar-sized; mixed sizes still work but waste compute on padding."""
        prepped = [self._prep(im, dpi) for im in images]
        Hm = max(((im.shape[0] + 31) // 32 * 32) for im in prepped)
        Wm = max(((im.shape[1] + 31) // 32 * 32) for im in prepped)
        xs, meta = [], []
        for im in prepped:
            h, w = im.shape; t, l = (Hm - h) // 2, (Wm - w) // 2
            p = cv.copyMakeBorder(im, t, Hm - h - t, l, Wm - w - l, cv.BORDER_CONSTANT, value=float(im[0, 0]))
            xs.append(p[None]); meta.append((l, t, w, h))
        x = torch.tensor(np.stack(xs).astype(np.float32), device=self.device)
        pos, dir2, typ = self._infer(x)
        scale = dpi / 500 if dpi != 500 else 1.0
        per = self.model.extract(pos, dir2, typ, q=quality)
        return [self._decode(per[i], *meta[i], scale) for i in range(len(prepped))]


__all__ = ["MinutiaeExtractor"]
