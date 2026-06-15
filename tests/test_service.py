"""FastAPI service tests. Skipped automatically if fastapi's TestClient deps (httpx) are absent.

Forces the extractor onto CPU before the service module imports it, so these run without a GPU.
"""
import os

import cv2 as cv
import numpy as np
import pytest

os.environ.setdefault("LEADER_DEVICE", "cpu")   # honored by service if it reads it; harmless otherwise

starlette_testclient = pytest.importorskip("starlette.testclient")


@pytest.fixture(scope="module")
def client(monkeypatch_session=None):
    # Build the extractor on CPU regardless of host, then import the app.
    import leader
    _orig = leader.MinutiaeExtractor

    def _cpu_extractor(*a, **k):
        k.setdefault("device", "cpu")
        k.setdefault("half", False)
        return _orig(*a, **k)

    leader.MinutiaeExtractor = _cpu_extractor
    try:
        from service.app import app
    finally:
        leader.MinutiaeExtractor = _orig
    return starlette_testclient.TestClient(app)


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
    assert "device" in body


def test_extract_endpoint(client):
    r = client.post("/extract", files={"file": ("latent.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == "latent.png"
    assert body["count"] == len(body["minutiae"])
    assert isinstance(body["minutiae"], list)


def test_extract_batch_endpoint(client):
    files = [("files", ("a.png", _png_bytes(), "image/png")),
             ("files", ("b.png", _png_bytes(), "image/png"))]
    r = client.post("/extract_batch", files=files)
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert results[0]["count"] == results[1]["count"]   # identical inputs → identical counts
