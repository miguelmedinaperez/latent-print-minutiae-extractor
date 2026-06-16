"""Test-time augmentation (`MinutiaeExtractor(tta=True)`) — averaging the detection map over 4 flips.

Includes a regression test for the CUDA-graph buffer-reuse bug: with `compile=True` the network's
outputs are views into a static replay buffer that the *next* forward overwrites, so the TTA
accumulator (which keeps maps alive across 5 forwards) must clone each map out of that buffer.
"""
import numpy as np
import torch

from leader import MinutiaeExtractor


def _bare_extractor(tta):
    """A MinutiaeExtractor with only the attributes `_infer` touches — no weight load."""
    ex = MinutiaeExtractor.__new__(MinutiaeExtractor)
    ex.tta = tta
    return ex


def test_tta_averages_four_flips():
    """Identity forward → constant pos map; the 4-flip average must equal that same constant, and the
    model is run 5 times (4 flips + 1 identity pass for dir/typ)."""
    ex = _bare_extractor(tta=True)
    calls = {"n": 0}

    def const_forward(x):
        calls["n"] += 1
        b, _, h, w = x.shape
        pos = torch.full((b, 1, h, w), 0.7)
        dir2 = torch.zeros(b, 2, h, w)
        typ = torch.full((b, 1, h, w), 0.8)
        return pos, dir2, typ

    ex._forward = const_forward
    pos, dir2, typ = ex._infer(torch.zeros(1, 1, 8, 8))
    assert calls["n"] == 5
    assert torch.allclose(pos, torch.full_like(pos, 0.7))


def test_tta_clones_detection_map_across_forwards():
    """Regression for the CUDA-graph reuse error
    ("accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run").

    We simulate a CUDA-graph: a single shared buffer that every forward overwrites in place and then
    hands back as a view. Per-call pos values are 0,1,2,3, so the correct TTA mean is 1.5. Without
    cloning, the no-flip term aliases the shared buffer and reads a later write (→ 1.75); the clone in
    `_infer` makes each term own its data, giving the correct 1.5. (On real CUDA the un-cloned path
    raises instead of silently corrupting — same root cause, same fix.)"""
    ex = _bare_extractor(tta=True)
    H = W = 8
    shared = torch.zeros(1, 1, H, W)      # mimics the CUDA-graph static output buffer
    calls = {"n": 0}

    def aliasing_forward(x):
        shared.fill_(float(calls["n"]))   # overwrite the buffer in place ...
        calls["n"] += 1
        dir2 = torch.zeros(1, 2, H, W)
        typ = torch.full((1, 1, H, W), 0.8)
        return shared, dir2, typ          # ... and return a *view* of it, not a fresh tensor

    ex._forward = aliasing_forward
    pos, _, _ = ex._infer(torch.zeros(1, 1, H, W))

    assert calls["n"] == 5
    assert torch.allclose(pos, torch.full_like(pos, 1.5)), pos.unique()


def test_no_tta_is_single_pass():
    """Without TTA the model is invoked exactly once and its outputs are passed straight through."""
    ex = _bare_extractor(tta=False)
    calls = {"n": 0}

    def fwd(x):
        calls["n"] += 1
        b, _, h, w = x.shape
        return torch.full((b, 1, h, w), 0.3), torch.zeros(b, 2, h, w), torch.zeros(b, 1, h, w)

    ex._forward = fwd
    pos, _, _ = ex._infer(torch.zeros(1, 1, 8, 8))
    assert calls["n"] == 1
    assert torch.allclose(pos, torch.full_like(pos, 0.3))
