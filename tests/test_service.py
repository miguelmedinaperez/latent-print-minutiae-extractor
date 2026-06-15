"""FastAPI service tests. Skipped automatically if starlette's TestClient deps (httpx) are absent.

The service's extractor is forced onto CPU and its forward is replaced by a single known detection
peak (see conftest.inject_one_peak_forward), so the endpoint assertions check a real, non-empty,
deterministic result instead of passing vacuously on an empty list.
"""
import cv2 as cv
import numpy as np
import pytest

from conftest import inject_one_peak_forward

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient   # noqa: E402


@pytest.fixture(scope="module")
def client():
    import leader
    _orig = leader.MinutiaeExtractor

    def _cpu_extractor(*a, **k):
        k.setdefault("device", "cpu")
        k.setdefault("half", False)
        return _orig(*a, **k)

    leader.MinutiaeExtractor = _cpu_extractor
    try:
        from service.app import app
        import service.app as appmod
        inject_one_peak_forward(appmod.extractor)        # deterministic single-peak forward
    finally:
        leader.MinutiaeExtractor = _orig
    return TestClient(app)


def _png_bytes():
    h, w = 160, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.clip(128 + 90 * np.sin(xx / 6.0), 0, 255).astype(np.uint8)
    ok, buf = cv.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


def test_extract_endpoint(client):
    r = client.post("/extract", files={"file": ("latent.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == "latent.png"
    assert body["count"] == 1                            # injected single peak → non-vacuous
    assert body["count"] == len(body["minutiae"])
    assert set(body["minutiae"][0]) == {"x", "y", "angle", "quality"}


def test_extract_batch_endpoint(client):
    files = [("files", ("a.png", _png_bytes(), "image/png")),
             ("files", ("b.png", _png_bytes(), "image/png"))]
    r = client.post("/extract_batch", files=files)
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert results[0]["count"] == 1 and results[1]["count"] == 1
    assert results[0]["minutiae"] == results[1]["minutiae"]   # identical inputs → identical output
