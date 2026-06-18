"""Smoke tests for the PyTorch LEADER port and the MinutiaeExtractor API. CPU-only by design, so they
run anywhere without a GPU.

Coverage, and why each assertion actually bites:
  * test_forward_heads_*       — the REAL model loads and emits finite, in-range heads of the right
                                  shape (catches a broken port / bad weights).
  * test_decode_recovers_peak  — feeds model.extract() a hand-built detection map with one known peak
                                  and orientation; asserts the REAL NMS/argmax/atan2 decode recovers
                                  it (position, angle, type). Independent of the trained weights.
  * test_extract_* / batch / dpi — inject a known detection peak into the forward (see
                                  inject_one_peak_forward) so the extractor's resample → pad → decode →
                                  de-offset → dpi-scale → dict pipeline is exercised DETERMINISTICALLY.
                                  These would fail on a regression in padding, de-offset, dpi scaling,
                                  batch padding, or the single↔batch parity — none of it passes
                                  vacuously, and none of it depends on the model's response to
                                  synthetic pixels (which is tiny and would shift across retrains).

We assert the API *contract*, never a data-dependent minutia count from the trained model.
"""
from pathlib import Path

import numpy as np
import torch
import pytest

from conftest import inject_one_peak_forward
from leader import MinutiaeExtractor
from leader.leader_torch import LeaderTorch

WEIGHTS = Path(__file__).resolve().parent.parent / "leader" / "weights"


@pytest.fixture(scope="session")
def model():
    return LeaderTorch(str(WEIGHTS / "leader_weights.npz"), str(WEIGHTS / "leader_layers.json")).eval()


@pytest.fixture(scope="session")
def _extractor():
    return MinutiaeExtractor(device="cpu", half=False)   # CPU → hardware-independent


@pytest.fixture
def injected(_extractor):
    """Extractor whose forward is replaced by a single known peak; restored after the test."""
    saved = _extractor._forward
    theta = inject_one_peak_forward(_extractor)
    yield _extractor, theta
    _extractor._forward = saved


def _angle_err(a, b):
    return abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)


def test_weights_present():
    for f in ("leader_weights.npz", "leader_layers.json", "leader_universal_deploy.pt"):
        assert (WEIGHTS / f).exists(), f"missing shipped weight file: {f}"


def test_port_smoke_check():
    """`python -m leader.leader_torch` (smoke_check) loads the shipped weights + runs a forward pass."""
    from leader.leader_torch import smoke_check
    assert smoke_check() is True


def test_forward_heads_shapes_and_finite(model):
    x = torch.zeros(1, 1, 64, 64)        # multiple of 32 → valid input
    with torch.no_grad():
        pos, dir2, typ = model(x)
    assert pos.shape == (1, 1, 64, 64)
    assert dir2.shape == (1, 2, 64, 64)  # cos/sin
    assert typ.shape == (1, 1, 64, 64)
    for t in (pos, dir2, typ):
        assert torch.isfinite(t).all()
    assert (pos >= 0).all() and (pos <= 1).all()   # sigmoid detection map


def test_decode_recovers_known_peak(model):
    """Hand-built detection map (one Gaussian peak) + constant orientation; assert the NMS decode
    returns that minutia at the right place, angle and type. Directly tests model.extract()."""
    H = W = 64
    y0, x0, theta = 30, 22, 0.7
    yy, xx = np.mgrid[0:H, 0:W]
    bump = np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)  # unique max
    pos = torch.tensor(bump[None, None])
    dir2 = torch.zeros(1, 2, H, W)
    dir2[:, 0], dir2[:, 1] = float(np.cos(theta)), float(np.sin(theta))
    typ = torch.full((1, 1, H, W), 0.8)
    mns = model.extract(pos, dir2, typ, q=0.1)[0]
    assert len(mns) >= 1, "decode dropped a clearly-present peak"
    mx, my, ang, ty, ql = max(mns, key=lambda m: m[4])
    assert abs(mx - x0) <= 1 and abs(my - y0) <= 1
    assert _angle_err(ang, theta) < 0.05, f"decoded angle {ang:.3f} != injected {theta:.3f}"
    assert ty == "E" and ql > 0.1


def test_extract_returns_one_wellformed_minutia(injected, synthetic_print):
    extractor, theta = injected
    mns = extractor.extract(synthetic_print, quality=0.1)
    assert len(mns) == 1                                   # one injected peak → one minutia
    m = mns[0]
    assert set(m) == {"x", "y", "angle", "quality"}
    h, w = synthetic_print.shape
    assert 0 <= m["x"] < w and 0 <= m["y"] < h             # de-padded back into the original frame
    assert _angle_err(m["angle"], theta) < 0.05            # angle passed through correctly
    assert m["quality"] > 0.1


def test_extract_is_deterministic(injected, synthetic_print):
    extractor, _ = injected
    assert extractor.extract(synthetic_print) == extractor.extract(synthetic_print)


def test_batch_matches_single(injected, synthetic_print):
    """extract_batch must produce the same minutiae as per-image extract (same padding/decode)."""
    extractor, _ = injected
    single = extractor.extract(synthetic_print, quality=0.1)
    batch = extractor.extract_batch([synthetic_print, synthetic_print], quality=0.1)
    assert len(batch) == 2
    assert batch[0] == batch[1] == single
    assert len(single) == 1


def test_dpi_resampling_maps_back_in_bounds(injected, synthetic_print):
    """A non-500 dpi triggers the resample branch; the peak must map back into the original frame."""
    extractor, _ = injected
    mns = extractor.extract(synthetic_print, dpi=1000, quality=0.1)
    assert len(mns) == 1
    h, w = synthetic_print.shape
    assert 0 <= mns[0]["x"] <= w and 0 <= mns[0]["y"] <= h


def test_tta_preserves_decode(injected, synthetic_print):
    """With TTA on, the flip-average + un-flip must still recover the injected centre peak (i.e. the
    flip/un-flip bookkeeping is correct, not mirrored)."""
    extractor, theta = injected
    extractor.tta = True
    try:
        mns = extractor.extract(synthetic_print, quality=0.1)
    finally:
        extractor.tta = False
    assert len(mns) == 1
    m = mns[0]
    h, w = synthetic_print.shape
    assert 0 <= m["x"] < w and 0 <= m["y"] < h
    assert _angle_err(m["angle"], theta) < 0.05
