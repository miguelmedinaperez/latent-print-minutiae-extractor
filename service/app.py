"""FastAPI minutiae-extraction service. One model loaded per worker/pod; stateless, so it scales
horizontally behind a load balancer / Kubernetes Deployment + HPA.

    uvicorn service.app:app --host 0.0.0.0 --port 8000
    curl -F file=@latent.png "http://localhost:8000/extract?dpi=500&quality=0.1"
"""
from fastapi import FastAPI, UploadFile, File, Query
import numpy as np
import cv2 as cv
from leader import MinutiaeExtractor

app = FastAPI(title="Latent Print Minutiae Extractor", version="1.0")
extractor = MinutiaeExtractor()   # loaded once at import (per worker)


def _decode(buf):
    img = cv.imdecode(np.frombuffer(buf, np.uint8), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "device": extractor.device}


@app.post("/extract")
async def extract(file: UploadFile = File(...), dpi: int = Query(500), quality: float = Query(0.1)):
    """Extract minutiae from one fingerprint or palmprint."""
    mns = extractor.extract(_decode(await file.read()), dpi=dpi, quality=quality)
    return {"image": file.filename, "count": len(mns), "minutiae": mns}


@app.post("/extract_batch")
async def extract_batch(files: list[UploadFile] = File(...), dpi: int = Query(500), quality: float = Query(0.1)):
    """Extract from several images in one forward pass (best for similar-sized prints)."""
    imgs = [_decode(await f.read()) for f in files]
    results = extractor.extract_batch(imgs, dpi=dpi, quality=quality)
    return {"results": [{"image": f.filename, "count": len(m), "minutiae": m} for f, m in zip(files, results)]}
