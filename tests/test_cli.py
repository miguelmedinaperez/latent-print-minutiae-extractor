"""CLI tests for `python -m leader.infer`. We monkeypatch the extractor with a known single-peak
forward (see conftest.inject_one_peak_forward) so the TSV/JSON output is deterministic and the
assertions actually bite (count, column layout, batch parity) — none of it depends on the trained
model's response to synthetic pixels."""
import sys
import cv2 as cv
import pytest

from conftest import inject_one_peak_forward
import leader.infer as infer
from leader import MinutiaeExtractor


@pytest.fixture
def injected_extractor():
    ex = MinutiaeExtractor(device="cpu", half=False)
    inject_one_peak_forward(ex)            # -> exactly one minutia per image
    return ex


def _patch(monkeypatch, ex, argv):
    monkeypatch.setattr(infer, "MinutiaeExtractor", lambda *a, **k: ex)
    monkeypatch.setattr(sys, "argv", argv)


def test_cli_single_tsv(tmp_path, monkeypatch, injected_extractor, synthetic_print):
    img = tmp_path / "p.png"; cv.imwrite(str(img), synthetic_print)
    out = tmp_path / "m.tsv"
    _patch(monkeypatch, injected_extractor, ["infer", str(img), "--out", str(out), "--quality", "0.1"])
    infer.main()
    lines = out.read_text().splitlines()
    assert len(lines) == 1                                   # one injected peak -> one row
    cols = lines[0].split("\t")
    assert len(cols) == 4                                    # x  y  angle(rad)  quality
    x, y, ang, q = (float(c) for c in cols)
    assert x.is_integer() and y.is_integer() and 0.0 <= q <= 1.0


def test_cli_batch_out_dir(tmp_path, monkeypatch, injected_extractor, synthetic_print):
    for nm in ("a", "b"):
        cv.imwrite(str(tmp_path / f"{nm}.png"), synthetic_print)
    outdir = tmp_path / "out"
    _patch(monkeypatch, injected_extractor,
           ["infer", str(tmp_path / "*.png"), "--batch", "--out-dir", str(outdir)])
    infer.main()
    tsvs = sorted(outdir.glob("*.tsv"))
    assert [p.name for p in tsvs] == ["a.tsv", "b.tsv"]
    for p in tsvs:
        assert len(p.read_text().splitlines()) == 1


def test_cli_json(tmp_path, monkeypatch, injected_extractor, synthetic_print, capsys):
    img = tmp_path / "p.png"; cv.imwrite(str(img), synthetic_print)
    _patch(monkeypatch, injected_extractor, ["infer", str(img), "--json"])
    infer.main()
    import json
    rec = json.loads(capsys.readouterr().out.strip())
    assert rec["image"].endswith("p.png") and len(rec["minutiae"]) == 1
    assert set(rec["minutiae"][0]) == {"x", "y", "angle", "quality"}
