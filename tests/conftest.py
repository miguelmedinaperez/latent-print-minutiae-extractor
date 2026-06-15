"""Make the repo root importable (so `import leader` / `import service` work without installing),
and provide shared fixtures/helpers for the smoke tests."""
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_print():
    """A deterministic ridge-like grayscale image (not a real print). Its size is intentionally NOT a
    multiple of 32 so the extractor's padding / de-offset logic is exercised. Its pixel content is
    irrelevant to the contract tests, which inject the detection map (see inject_one_peak_forward)."""
    h, w = 200, 168
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.clip(128 + 90 * np.sin(xx / 6.0), 0, 255).astype(np.uint8)


def inject_one_peak_forward(extractor, theta=0.6, sigma=2.0):
    """Replace `extractor._forward` with a synthetic head producing ONE Gaussian detection peak at the
    centre of whatever (padded) input it receives, plus a constant orientation field (angle=theta) and
    type≈"E". This makes the surrounding pipeline — resample, pad32, the REAL NMS decode
    (model.extract), de-offset and dpi-scaling — deterministic and independent of the trained weights,
    so the contract tests don't depend on how the model happens to respond to synthetic pixels.
    Returns the injected angle. Caller is responsible for restoring `extractor._forward`."""
    def fake_forward(x):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
        bump = torch.exp(-((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (2 * sigma ** 2))  # unique max
        pos = bump[None, None].repeat(b, 1, 1, 1)
        dir2 = torch.zeros(b, 2, h, w)
        dir2[:, 0], dir2[:, 1] = float(np.cos(theta)), float(np.sin(theta))
        typ = torch.full((b, 1, h, w), 0.8)
        return pos, dir2, typ
    extractor._forward = fake_forward
    return theta
