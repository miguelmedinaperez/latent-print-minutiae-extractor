"""minutiae_superset — confidence-ranked superset minutiae extractor in PyTorch.

Runs FingerNet and MinutiaeNet (CoarseNet + FineNet) — both ported from their
MIT-licensed Keras/TF1 originals and verified to reproduce the reference outputs —
and fuses them into an agreement-weighted superset. See the package README and NOTICE
for attribution and licensing.
"""
from .extractor import MinutiaeExtractor, extract_minutiae

__all__ = ["MinutiaeExtractor", "extract_minutiae"]
__version__ = "0.1.0"
