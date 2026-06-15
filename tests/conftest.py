"""Make the repo root importable (so `import leader` / `import service` work without installing),
and provide a shared synthetic latent-print fixture used across the smoke tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_print():
    """A deterministic ridge-like grayscale image (not a real print — just structured input so the
    extractor exercises its full decode path). Size is intentionally NOT a multiple of 32 to test
    the padding logic."""
    h, w = 200, 168
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ridges = 128 + 90 * np.sin(xx / 6.0 + 4.0 * np.sin(yy / 40.0))   # curved ridge flow
    return np.clip(ridges, 0, 255).astype(np.uint8)
