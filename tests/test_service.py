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
    assert set(body) >= {"status", "device", "tta", "compiled", "half"}   # reports the config
    assert body["tta"] is False and body["compiled"] is False


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


def test_configure(client):
    """/configure (re)initializes the extractor and is where tta/compile are set."""
    import service.app as appmod
    r = client.post("/configure", json={"tta": True, "compile": False})
    assert r.status_code == 200
    body = r.json()
    assert body["tta"] is True and body["compiled"] is False        # compile is GPU-only; CPU host → False
    assert body["device"] in ("cpu", "cuda") and "half" in body
    # /configure rebuilt the extractor with a real forward; re-inject the deterministic one and confirm
    # the configured TTA is honoured by /extract (the flip-symmetric injected peak survives averaging).
    inject_one_peak_forward(appmod.extractor)
    r2 = client.post("/extract", files={"file": ("latent.png", _png_bytes(), "image/png")})
    assert r2.status_code == 200 and r2.json()["count"] == 1
    assert appmod.extractor.tta is True


def test_plot_endpoint_tsv(client):
    tsv = b"40\t60\t0.5\t0.9\n80\t90\t-1.2\t0.4\n"
    r = client.post("/plot", files={"file": ("latent.png", _png_bytes(), "image/png"),
                                    "minutiae": ("m.tsv", tsv, "text/tab-separated-values")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"          # valid PNG magic
    assert len(r.content) > 1000


def test_plot_endpoint_json(client):
    import json
    body = json.dumps([{"x": 40, "y": 60, "angle": 0.5, "quality": 0.9}]).encode()
    r = client.post("/plot", files={"file": ("latent.png", _png_bytes(), "image/png"),
                                    "minutiae": ("m.json", body, "application/json")})
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
