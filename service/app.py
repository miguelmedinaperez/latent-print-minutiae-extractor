"""FastAPI minutiae-extraction service. One model loaded per worker/pod; stateless across requests,
so it scales horizontally behind a load balancer / Kubernetes Deployment + HPA.

    uvicorn service.app:app --host 0.0.0.0 --port 8000

Endpoints (interactive docs at /docs, OpenAPI at /openapi.json):
  GET  /health                            -> JSON {status, device, tta, compiled, half}
  POST /configure  {tta, compile, half}   -> JSON config  ((re)initialize the extractor — the HTTP form
                                             of MinutiaeExtractor(tta=, compile=, half=))
  POST /extract        file [?dpi&quality&format]   -> the minutiae FILE (text/tab-separated-values) by
                                                default, or JSON when `?format=json`; count in the
                                                `X-Minutiae-Count` header.
  POST /extract_batch  files[] [?dpi&quality&format] -> application/zip of one `<image-stem>.tsv` per image
                                                (default), or JSON `{results:[...]}` when `?format=json`.
  POST /plot           file + minutiae(file)  -> the overlay PNG (markers on the confidence scale).

Output format (the minutiae file). Tab-separated, a **header row** then one minutia per line — identical to
`leader.infer` / the `leader-extract` CLI, so /extract output feeds straight back into /plot:
    x   y   angle   quality   type
  * x, y   — integer pixel coordinates in the INPUT image (origin top-left, x→right, y→down).
  * angle  — ridge direction in radians, range (-pi, pi], in LEADER's native convention. To draw it or
             compare against the usual examiner convention on an image (y down), NEGATE it. See the README
             "Output format & angle convention".
  * quality — detection confidence in (0, 1]; the `quality` query arg drops anything below it.
  * type    — "E" (ending) / "B" (bifurcation), from LEADER's type head (inherited from pretraining, not
             re-validated on latents — best-effort). JSON minutiae carry the same five fields.

`tta` and `compile` are extractor-construction settings, so they are set ONCE via /configure
(`compile` = CUDA graphs, GPU-only, ~1 min warmup per input size), not per request. `dpi`/`quality`
are per-request.
"""
import io, json, os, zipfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2 as cv
from leader import MinutiaeExtractor
from leader.infer import _tsv          # the exact TSV formatter the CLI uses (byte-identical output)
from leader.viz import overlay_png

# OpenAPI example of the TSV body (header row + x y angle quality type), so /docs shows the real shape.
_TSV_EXAMPLE = "x\ty\tangle\tquality\ttype\n238\t512\t-1.972921\t0.731000\tB\n301\t140\t0.845210\t0.402000\tE\n"


def _bool_env(name):
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


app = FastAPI(title="Latent Print Minutiae Extractor", version="1.3")
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
    """Parse an uploaded minutiae file: whitespace/TSV `x y angle [quality] [type]` (a header row and
    blank/`#` lines are skipped), a JSON list of {x,y,angle,quality,type}, or a JSON object with a
    `minutiae` list. Returns a list of dicts."""
    text = buf.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    if filename.lower().endswith(".json") or text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("minutiae", [])
        return [{"x": float(m["x"]), "y": float(m["y"]), "angle": float(m["angle"]),
                 "quality": float(m.get("quality", 1.0)), "type": m.get("type", "")} for m in data]
    out = []
    for ln in text.splitlines():
        p = ln.split()
        if len(p) < 3:
            continue
        try:
            x, y, ang = float(p[0]), float(p[1]), float(p[2])
        except ValueError:
            continue                                   # skip the header row / comments / non-numeric lines
        out.append({"x": x, "y": y, "angle": ang,
                    "quality": float(p[3]) if len(p) >= 4 else 1.0,
                    "type": p[4] if len(p) >= 5 else ""})
    return out


class Config(BaseModel):
    tta: bool = False        # average the detection map over 4 flips (~+0.02 AP, ~5x forward cost)
    compile: bool = False    # CUDA-graph compile (~2.5x faster on GPU; ~1 min warmup per input size; no-op on CPU)
    half: bool | None = None  # fp16 on GPU (None = auto: on for CUDA, off for CPU)


@app.get("/health")
def health():
    """Readiness + the current extractor configuration (JSON: status, device, tta, compiled, half)."""
    return {"status": "ok", **_config()}


@app.post("/configure")
def configure(cfg: Config):
    """(Re)initialize the extractor — the HTTP equivalent of
    `MinutiaeExtractor(tta=..., compile=..., half=...)`. This is where `tta` and `compile` are set
    (they are construction settings, not per-request flags). Returns the resulting config as JSON."""
    global extractor
    extractor = MinutiaeExtractor(tta=cfg.tta, compile=cfg.compile, half=cfg.half)
    return _config()


@app.post("/extract", response_class=Response, responses={200: {
    "content": {"text/tab-separated-values": {"example": _TSV_EXAMPLE},
                "application/json": {"example": {"image": "latent.png", "count": 1, "minutiae": [
                    {"x": 238, "y": 512, "angle": -1.972921, "quality": 0.731, "type": "B"}]}}},
    "description": "By default the minutiae TSV file (header row + `x y angle quality type`, count in the "
                   "`X-Minutiae-Count` header, download name in `Content-Disposition`); JSON when "
                   "`?format=json`."}})
async def extract(file: UploadFile = File(..., description="one fingerprint/palmprint image (any format OpenCV reads)"),
                  dpi: int = Query(500, description="scan resolution of the input image"),
                  quality: float = Query(0.1, description="detection threshold in (0,1]; lower = more minutiae"),
                  format: str = Query("tsv", pattern="^(tsv|json)$", description="response format: tsv file (default) or json")):
    """Extract minutiae from one fingerprint or palmprint.

    **Input:** an image file (multipart `file`); query `dpi`, `quality`, `format` (`tsv`|`json`). Uses the
    configured extractor (set TTA / compile via /configure).

    **Output (default `format=tsv`):** the minutiae **file** — `text/tab-separated-values` with a header
    row then one minutia per line `x  y  angle  quality  type` (x,y in input pixels; angle in radians in
    LEADER's convention — negate to draw/compare; quality in (0,1]; type "E"/"B"). The count is in the
    `X-Minutiae-Count` header. Byte-identical to `leader-extract --out`, and feeds straight into /plot.
    **`format=json`:** `{image, count, minutiae:[{x,y,angle,quality,type}, ...]}`."""
    mns = extractor.extract(_decode(await file.read()), dpi=dpi, quality=quality)
    if format == "json":
        return JSONResponse({"image": file.filename, "count": len(mns), "minutiae": mns})
    stem = Path(file.filename or "minutiae").stem
    return Response(content=_tsv(mns), media_type="text/tab-separated-values",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.tsv"',
                             "X-Minutiae-Count": str(len(mns))})


@app.post("/extract_batch", response_class=Response, responses={200: {
    "content": {"application/zip": {}, "application/json": {}},
    "description": "A ZIP of one `<image-stem>.tsv` per image (default), or JSON "
                   "`{results:[{image,count,minutiae}]}` when `?format=json`."}})
async def extract_batch(files: list[UploadFile] = File(..., description="several images, processed in one pass"),
                        dpi: int = Query(500), quality: float = Query(0.1),
                        format: str = Query("zip", pattern="^(zip|json)$", description="response format: zip of TSVs (default) or json")):
    """Extract from several images in one forward pass (best for similar-sized prints).

    **Input:** multipart `files[]`; query `dpi`, `quality`, `format` (`zip`|`json`).
    **Output (default):** `application/zip` — one `<image-stem>.tsv` per image (same TSV format as
    /extract). **`format=json`:** `{results:[{image, count, minutiae:[...]}, ...]}`."""
    imgs = [_decode(await f.read()) for f in files]
    results = extractor.extract_batch(imgs, dpi=dpi, quality=quality)
    if format == "json":
        return JSONResponse({"results": [{"image": f.filename, "count": len(m), "minutiae": m}
                                         for f, m in zip(files, results)]})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f, mns in zip(files, results):
            z.writestr(f"{Path(f.filename or 'image').stem}.tsv", _tsv(mns))
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="minutiae.zip"'})


@app.post("/plot", response_class=Response, responses={200: {
    "content": {"image/png": {}}, "description": "The overlay PNG."}})
async def plot(file: UploadFile = File(..., description="the print image the minutiae were extracted from"),
               minutiae: UploadFile = File(..., description="a minutiae file from /extract or leader-extract (TSV), or JSON")):
    """Overlay an uploaded minutiae file on its print and return a PNG.

    **Input:** the print image (`file`) + a minutiae file (`minutiae`) — the TSV from /extract /
    `leader-extract`, or a JSON list/object.
    **Output:** an `image/png` overlay — a hollow circle + a short direction line per minutia, coloured
    by the **confidence scale** (RdYlGn, with a colorbar)."""
    img = _decode(await file.read())
    mns = _parse_minutiae(await minutiae.read(), minutiae.filename or "")
    png = overlay_png(img, mns, f"{file.filename} — {len(mns)} minutiae")
    return Response(content=png, media_type="image/png")


if __name__ == "__main__":      # debug entrypoint: `python -m service.app` (or F5 — see .vscode/launch.json)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
