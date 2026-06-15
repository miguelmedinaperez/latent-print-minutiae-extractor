"""Confidence-ranked superset minutiae extractor (FingerNet ∪ MinutiaeNet).

The "new algorithm": run both detectors and output every minutia, ranked by an
agreement-weighted confidence —
  * detected by BOTH (centres within ``match_dist`` px @500dpi) -> mn_score + fn_score
  * detected by ONE only                                       -> that detector's score
so agreement ranks highest and single-detector hits rank by their own score.

Designed for 500-dpi imagery but imposes NO size limit: other DPIs are resampled to
the 500-dpi ridge scale, and arbitrarily large inputs (e.g. palmprints) are processed
by overlapping tiles, then merged. The detectors were trained on fingerprints; on
palmprints the pipeline RUNS but detection quality is unvalidated.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .fingernet import FingerNet, extract as _fingernet_extract, nms as _nms
from .coarsenet import CoarseNet
from .finenet import FineNet
from .minutiaenet import extract as _minutiaenet_extract

_WEIGHTS = Path(__file__).resolve().parent / "weights"


def _to_gray_u8(image):
    """Accept a path / PIL.Image / ndarray (gray or RGB) -> HxW uint8 grayscale."""
    import cv2
    if isinstance(image, (str, Path)):
        arr = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"cannot read image: {image}")
        return arr
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _tile_starts(n, tile, overlap):
    step = tile - overlap
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, step))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


class MinutiaeExtractor:
    """Loads both detectors once; call :meth:`extract` per image."""

    def __init__(self, weights_dir=None, device=None, match_dist=15.0,
                 max_tile=1024, tile_overlap=128, conf_scale=0.5,
                 refine_orientation=True, gpu_orientation=None, use_finenet=True,
                 match_angle="fingernet", mn_max_minu=20, mn_min_minu=6,
                 fusion="union"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.match_dist = float(match_dist)
        # For agreement-matched ("both") minutiae, whose angle to report. FingerNet's
        # angles are well-aligned to GT (~4deg median residual) while MinutiaeNet's carry
        # a pi-flip ambiguity, so "fingernet" markedly improves the location+angle metric
        # at zero cost to location-only (positions/scores unchanged). "minutiaenet" keeps
        # the original recipe.
        assert match_angle in ("fingernet", "minutiaenet")
        self.match_angle = match_angle
        # Fusion strategy:
        #   "union"     (default) — every FingerNet + MinutiaeNet minutia, agreement-weighted.
        #               Best on FINGERPRINTS, where both detectors are strong (MN-unique
        #               minutiae add recall).
        #   "reinforce" — FingerNet is the base set; MinutiaeNet only BOOSTS the confidence
        #               of minutiae the two agree on; NO MinutiaeNet-unique minutiae are
        #               added. Better when MinutiaeNet is weak/noisy (e.g. PALMPRINTS):
        #               keeps FingerNet's recall, uses agreement only to re-rank.
        assert fusion in ("union", "reinforce")
        self.fusion = fusion
        # MinutiaeNet's adaptive threshold targets min_minu..max_minu minutiae PER TILE.
        # The faithful fingerprint default (20/6) under-detects on dense palmprint tiles;
        # raise mn_max_minu for palmprints. Per-tile, so it scales with the number of tiles.
        self.mn_max_minu = int(mn_max_minu)
        self.mn_min_minu = int(mn_min_minu)
        self.max_tile = int(max_tile)
        self.tile_overlap = int(tile_overlap)
        # FineNet re-scores MinutiaeNet candidates (soft fusion). use_finenet=False skips
        # it (~faster) but CHANGES scores + which minutiae survive, so it can move the AP.
        self.use_finenet = bool(use_finenet)
        # MinutiaeNet's STFT ridge-flow orientation refinement is ~80% of the CPU
        # post-processing and only sets minutiae ANGLES (positions/scores/detections
        # are unaffected). Two speedups:
        #   refine_orientation=False -> skip it, use the network's orientation head (~4x)
        #   gpu_orientation=True     -> compute the STFT on the GPU (torch.fft, ~4x);
        #     angles match the CPU STFT to <0.5deg median (100% within 10deg).
        # gpu_orientation defaults to True on CUDA (the recommended fast+faithful config)
        # and False on CPU (numpy STFT). All AP/PR metrics are identical either way.
        self.refine_orientation = bool(refine_orientation)
        self.gpu_orientation = (self.device == "cuda") if gpu_orientation is None \
            else bool(gpu_orientation)
        # Per-detector scores are sigmoids in (0,1], so the raw agreement-weighted
        # confidence (mn+fn) spans (0,2]. conf_scale=0.5 maps it to (0,1] without
        # changing the ranking (monotonic), so all AP/PR metrics are unchanged.
        self.conf_scale = float(conf_scale)
        wd = Path(weights_dir) if weights_dir else _WEIGHTS

        self.fingernet = FingerNet()
        self.fingernet.load_state_dict(torch.load(wd / "fingernet.pt", map_location="cpu"))
        self.coarsenet = CoarseNet()
        self.coarsenet.load_state_dict(torch.load(wd / "coarsenet.pt", map_location="cpu"))
        self.finenet = FineNet()
        self.finenet.load_state_dict(torch.load(wd / "finenet.pt", map_location="cpu"))
        for m in (self.fingernet, self.coarsenet, self.finenet):
            m.to(self.device).eval()

    # ---- per-detector minutiae on a (possibly tiled) image ----------------
    def _run_detector(self, kind, img):
        h, w = img.shape
        if h <= self.max_tile and w <= self.max_tile:
            return self._detect(kind, img)
        out = []
        for y0 in _tile_starts(h, self.max_tile, self.tile_overlap):
            for x0 in _tile_starts(w, self.max_tile, self.tile_overlap):
                tile = img[y0:y0 + self.max_tile, x0:x0 + self.max_tile]
                m = self._detect(kind, tile)
                if len(m):
                    m = m.copy()
                    m[:, 0] += x0
                    m[:, 1] += y0
                    out.append(m)
        if not out:
            return np.zeros((0, 4))
        return _nms(np.concatenate(out, axis=0))      # dedup tile-overlap doubles

    def _detect(self, kind, tile):
        if kind == "fingernet":
            return _fingernet_extract(self.fingernet, tile, self.device)
        return _minutiaenet_extract(self.coarsenet, self.finenet, tile, self.device,
                                    use_finenet=self.use_finenet,
                                    refine_orientation=self.refine_orientation,
                                    gpu_orientation=self.gpu_orientation,
                                    max_minu=self.mn_max_minu, min_minu=self.mn_min_minu)

    # ---- public API -------------------------------------------------------
    @torch.no_grad()
    def extract(self, image, dpi=500, method="superset"):
        """Return a list of minutiae dicts sorted by descending confidence:
        ``{x, y, angle, confidence, source}`` in ORIGINAL-image pixel coordinates.
        ``confidence`` is in (0, 1] (agreement-weighted; ``both`` hits score highest),
        ``source`` is 'both' / 'minutiaenet' / 'fingernet', ``angle`` is radians.

        method: "superset" (default, FingerNet+MinutiaeNet fusion — best on fingerprints),
        "fingernet", or "minutiaenet". On PALMPRINTS, FingerNet alone is strongest
        (MinutiaeNet under-detects out-of-domain), so use method="fingernet"."""
        import cv2
        gray = _to_gray_u8(image)
        scale = 500.0 / float(dpi)
        if abs(scale - 1.0) > 1e-3:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            gray = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                     max(1, round(gray.shape[0] * scale))),
                              interpolation=interp)

        if method == "fingernet":
            det = self._run_detector("fingernet", gray)
            rows = [(r[0], r[1], r[2], r[3], "fingernet") for r in det]
        elif method == "minutiaenet":
            det = self._run_detector("minutiaenet", gray)
            rows = [(r[0], r[1], r[2], r[3], "minutiaenet") for r in det]
        else:
            fn = self._run_detector("fingernet", gray)
            mn = self._run_detector("minutiaenet", gray)
            rows = [(x, y, a, c * self.conf_scale, s) for x, y, a, c, s in self._superset(mn, fn)]

        inv = 1.0 / scale
        out = [{"x": float(x * inv), "y": float(y * inv), "angle": float(ang),
                "confidence": float(conf), "source": src} for x, y, ang, conf, src in rows]
        out.sort(key=lambda r: -r["confidence"])
        return out

    def _superset(self, mn, fn):
        """Agreement-weighted fusion -> rows (x, y, angle, confidence, source)."""
        from scipy.spatial.distance import cdist
        if self.fusion == "reinforce":
            # FingerNet base; MinutiaeNet only boosts agreement; no MN-unique minutiae.
            if not len(fn):
                return []
            if not len(mn):
                return [(r[0], r[1], r[2], r[3], "fingernet") for r in fn]
            d = cdist(fn[:, :2], mn[:, :2])
            nn_idx, nn_dist = d.argmin(axis=1), d.min(axis=1)
            rows = []
            for j in range(len(fn)):
                if nn_dist[j] <= self.match_dist:
                    rows.append((fn[j, 0], fn[j, 1], fn[j, 2],
                                 fn[j, 3] + mn[nn_idx[j], 3], "both"))
                else:
                    rows.append((fn[j, 0], fn[j, 1], fn[j, 2], fn[j, 3], "fingernet"))
            return rows
        rows = []
        if len(mn) and len(fn):
            nd = cdist(mn[:, :2], fn[:, :2])
            nn_idx, nn_dist = nd.argmin(axis=1), nd.min(axis=1)
            for i in range(len(mn)):
                if nn_dist[i] <= self.match_dist:
                    # position from MinutiaeNet; angle from FingerNet (cleaner) by default
                    ang = fn[nn_idx[i], 2] if self.match_angle == "fingernet" else mn[i, 2]
                    rows.append((mn[i, 0], mn[i, 1], ang,
                                 mn[i, 3] + fn[nn_idx[i], 3], "both"))
                else:
                    rows.append((mn[i, 0], mn[i, 1], mn[i, 2], mn[i, 3], "minutiaenet"))
            fn_to_mn = cdist(fn[:, :2], mn[:, :2]).min(axis=1)
            for j in range(len(fn)):
                if fn_to_mn[j] > self.match_dist:
                    rows.append((fn[j, 0], fn[j, 1], fn[j, 2], fn[j, 3], "fingernet"))
        elif len(mn):
            rows = [(r[0], r[1], r[2], r[3], "minutiaenet") for r in mn]
        else:
            rows = [(r[0], r[1], r[2], r[3], "fingernet") for r in fn]
        return rows


_DEFAULT = None


def extract_minutiae(image, dpi=500, **kwargs):
    """Convenience one-shot API using a cached default extractor.

    >>> from minutiae_superset import extract_minutiae
    >>> minutiae = extract_minutiae("latent.png", dpi=500)
    >>> minutiae[0]
    {'x': 326.0, 'y': 481.0, 'angle': 2.34, 'confidence': 0.97, 'source': 'both'}
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MinutiaeExtractor(**kwargs)
    return _DEFAULT.extract(image, dpi=dpi)
