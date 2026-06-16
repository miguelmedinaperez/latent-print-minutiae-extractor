"""FastAPI minutiae-extraction service. One model loaded per worker/pod; stateless, so it scales
horizontally behind a load balancer / Kubernetes Deployment + HPA.

    uvicorn service.app:app --host 0.0.0.0 --port 8000
    curl -F file=@latent.png "http://localhost:8000/extract?dpi=500&quality=0.1&tta=false"
    # overlay a minutiae file on its print -> PNG:
    curl -F file=@latent.png -F minutiae=@latent.tsv "http://localhost:8000/plot" -o overlay.png
"""
import json
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
import numpy as np
import cv2 as cv
from leader import MinutiaeExtractor
from leader.viz import overlay_png

app = FastAPI(title="Latent Print Minutiae Extractor", version="1.1")
extractor = MinutiaeExtractor()   # loaded once at import (per worker)


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


@app.get("/health")
def health():
    return {"status": "ok", "device": extractor.device}


@app.post("/extract")
async def extract(file: UploadFile = File(...), dpi: int = Query(500), quality: float = Query(0.1),
                  tta: bool = Query(False)):
    """Extract minutiae from one fingerprint or palmprint. `tta` = test-time augmentation (~+0.02 AP,
    ~5x cost)."""
    buf = await file.read()
    extractor.tta = tta                                   # set right before the sync extract (no await between)
    mns = extractor.extract(_decode(buf), dpi=dpi, quality=quality)
    return {"image": file.filename, "count": len(mns), "minutiae": mns}


@app.post("/extract_batch")
async def extract_batch(files: list[UploadFile] = File(...), dpi: int = Query(500),
                        quality: float = Query(0.1), tta: bool = Query(False)):
    """Extract from several images in one forward pass (best for similar-sized prints)."""
    imgs = [_decode(await f.read()) for f in files]
    extractor.tta = tta
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
