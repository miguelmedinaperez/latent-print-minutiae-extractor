"""Smoke tests for the PyTorch LEADER port and the MinutiaeExtractor API.

CPU-only by design, so they run in CI without a GPU. They check that the model loads, a forward pass
produces sane head tensors, extraction returns well-formed in-bounds minutiae, results are
deterministic, and single vs batch extraction agree. They do NOT assert a specific minutia count
(that is data-dependent); they assert the contract.
"""
from pathlib import Path

import numpy as np
import torch
import pytest

from leader import MinutiaeExtractor
from leader.leader_torch import LeaderTorch

WEIGHTS = Path(__file__).resolve().parent.parent / "leader" / "weights"


@pytest.fixture(scope="session")
def model():
    return LeaderTorch(str(WEIGHTS / "leader_weights.npz"), str(WEIGHTS / "leader_layers.json")).eval()


@pytest.fixture(scope="session")
def extractor():
    # force CPU so the test is hardware-independent
    return MinutiaeExtractor(device="cpu", half=False)


def test_weights_present():
    for f in ("leader_weights.npz", "leader_layers.json", "leader_universal_deploy.pt"):
        assert (WEIGHTS / f).exists(), f"missing shipped weight file: {f}"


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


def test_extract_returns_wellformed_minutiae(extractor, synthetic_print):
    mns = extractor.extract(synthetic_print, dpi=500, quality=0.1)
    assert isinstance(mns, list)
    h, w = synthetic_print.shape
    for m in mns:
        assert set(m) == {"x", "y", "angle", "quality"}
        assert 0 <= m["x"] < w and 0 <= m["y"] < h        # in-bounds after de-padding
        assert -np.pi - 1e-3 <= m["angle"] <= np.pi + 1e-3
        assert 0.0 <= m["quality"] <= 1.0


def test_extract_is_deterministic(extractor, synthetic_print):
    a = extractor.extract(synthetic_print)
    b = extractor.extract(synthetic_print)
    assert a == b


def test_batch_matches_single(extractor, synthetic_print):
    """extract_batch must produce the same minutiae as per-image extract (same padding/decode)."""
    single = extractor.extract(synthetic_print)
    batch = extractor.extract_batch([synthetic_print, synthetic_print])
    assert len(batch) == 2
    assert batch[0] == batch[1] == single


def test_dpi_resampling_runs(extractor, synthetic_print):
    # a non-500 dpi triggers the resample branch; coords should map back into the original frame
    mns = extractor.extract(synthetic_print, dpi=1000)
    h, w = synthetic_print.shape
    for m in mns:
        assert 0 <= m["x"] < w and 0 <= m["y"] < h
