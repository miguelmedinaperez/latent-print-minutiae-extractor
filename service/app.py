"""FastAPI minutiae-extraction service. One model loaded per worker/pod; stateless across requests,
so it scales horizontally behind a load balancer / Kubernetes Deployment + HPA.

    uvicorn service.app:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health                      -> {status, device, tta, compiled, half}
  POST /configure  {tta, compile, half}  -> (re)initialize the extractor (the HTTP form of
                                            MinutiaeExtractor(tta=, compile=, half=)); returns the config
  POST /extract        file [?dpi&quality]            -> {image, count, minutiae}
  POST /extract_batch  files[] [?dpi&quality]         -> {results: [...]}
  POST /plot           file + minutiae(file)          -> overlay PNG (markers on the confidence scale)

`tta` and `compile` are extractor-construction settings, so they are set ONCE via /configure
(`compile` = CUDA graphs, GPU-only, ~1 min warmup per input size), not per request. `dpi`/`quality`
are per-request.
"""
import json, os
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import cv2 as cv
from leader import MinutiaeExtractor
from leader.viz import overlay_png


def _bool_env(name):
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


app = FastAPI(title="Latent Print Minutiae Extractor", version="1.2")
# Startup config from env (so every pod is consistent in a multi-replica deployment); POST /configure
# overrides it at runtime on the worker that serves the request.
extractor = MinutiaeExtractor(tta=_bool_env("LEADER_TTA"), compile=_bool_env("LEADER_COMPILE"))


def _config():
    return {"device": extractor.device, "tta": extractor.tta,
            "compiled": extractor.compiled, "half": extractor.half}


def _decode(buf):
    img = cv.imdecode(np.frombuffer(buf, np.uint8), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image")
    return img


def _parse_minutiae(buf, filename=""):
    """Parse an uploaded minutiae file: TSV `x y angle quality` (quality optional), a JSON list of
    {x,y,angle,quality}, or the /extract JSON response object. Returns a list of dicts."""
    text = buf.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    if filename.lower().endswith(".json") or text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("minutiae", [])
        return [{"x": float(m["x"]), "y": float(m["y"]), "angle": float(m["angle"]),
                 "quality": float(m.get("quality", 1.0))} for m in data]
    out = []
    for ln in text.splitlines():
        p = ln.split()
        if len(p) >= 3:
            out.append({"x": float(p[0]), "y": float(p[1]), "angle": float(p[2]),
                        "quality": float(p[3]) if len(p) >= 4 else 1.0})
    return out


class Config(BaseModel):
    tta: bool = False        # average the detection map over 4 flips (~+0.02 AP, ~5x forward cost)
    compile: bool = False    # CUDA-graph compile (~2.5x faster on GPU; ~1 min warmup per input size; no-op on CPU)
    half: bool | None = None  # fp16 on GPU (None = auto: on for CUDA, off for CPU)


@app.get("/health")
def health():
    """Readiness + the current extractor configuration."""
    return {"status": "ok", **_config()}


@app.post("/configure")
def configure(cfg: Config):
    """(Re)initialize the extractor — the HTTP equivalent of
    `MinutiaeExtractor(tta=..., compile=..., half=...)`. This is where `tta` and `compile` are set
    (they are construction settings, not per-request flags). Returns the resulting config."""
    global extractor
    extractor = MinutiaeExtractor(tta=cfg.tta, compile=cfg.compile, half=cfg.half)
    return _config()


@app.post("/extract")
async def extract(file: UploadFile = File(...), dpi: int = Query(500), quality: float = Query(0.1)):
    """Extract minutiae from one fingerprint or palmprint (uses the configured extractor; set TTA via
    /configure)."""
    mns = extractor.extract(_decode(await file.read()), dpi=dpi, quality=quality)
    return {"image": file.filename, "count": len(mns), "minutiae": mns}


@app.post("/extract_batch")
async def extract_batch(files: list[UploadFile] = File(...), dpi: int = Query(500),
                        quality: float = Query(0.1)):
    """Extract from several images in one forward pass (best for similar-sized prints)."""
    imgs = [_decode(await f.read()) for f in files]
    results = extractor.extract_batch(imgs, dpi=dpi, quality=quality)
    return {"results": [{"image": f.filename, "count": len(m), "minutiae": m} for f, m in zip(files, results)]}


@app.post("/plot")
async def plot(file: UploadFile = File(...), minutiae: UploadFile = File(...)):
    """Overlay an uploaded minutiae file on its print and return a PNG: a hollow circle + a short
    direction line per minutia, coloured by the **confidence scale** (RdYlGn, with a colorbar). The
    minutiae file is the TSV (`x y angle quality`) from /extract or `leader.infer`, or JSON."""
    img = _decode(await file.read())
    mns = _parse_minutiae(await minutiae.read(), minutiae.filename or "")
    png = overlay_png(img, mns, f"{file.filename} — {len(mns)} minutiae")
    return Response(content=png, media_type="image/png")


if __name__ == "__main__":      # debug entrypoint: `python -m service.app` (or F5 — see .vscode/launch.json)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
