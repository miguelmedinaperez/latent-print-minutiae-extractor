"""Smoke test for `python -m leader.finetune` — a 1-epoch CPU run on two tiny synthetic
image+XML samples (small crop so it's fast). Asserts it produces a loadable fine-tuned state_dict.
Exercises read_gt, heatmap, sample_crop, the unfreeze/optimiser setup, the train loop, and save."""
import sys
import numpy as np
import cv2 as cv
import torch

import leader.finetune as ft


def _write_sample(d, name, h=128, w=128):
    rng = np.random.RandomState(abs(hash(name)) % 2**32)
    cv.imwrite(str(d / f"{name}.bmp"), (rng.rand(h, w) * 255).astype(np.uint8))
    mins = ((40, 50, 0.5), (80, 90, 1.2), (60, 30, 2.7))
    body = "".join(f'<Minutia X="{x}" Y="{y}" Angle="{a}"/>' for x, y, a in mins)
    (d / f"{name}.xml").write_text(f"<Document>{body}</Document>")


def test_finetune_one_epoch_writes_loadable_pt(tmp_path, monkeypatch):
    data = tmp_path / "db"; data.mkdir()
    _write_sample(data, "s1"); _write_sample(data, "s2")
    out = tmp_path / "ft.pt"
    monkeypatch.setattr(sys, "argv",
                        ["finetune", "--data", str(data), "--out", str(out),
                         "--epochs", "1", "--crop", "96", "--batch", "2"])
    ft.main()
    assert out.exists(), "fine-tune did not write the weights"
    sd = torch.load(str(out), map_location="cpu")
    assert isinstance(sd, dict) and len(sd) > 0          # a real state_dict


def test_finetune_helpers(tmp_path):
    """read_gt parses X/Y (decimal-comma tolerated); heatmap is a [0,1] map peaking at the GT."""
    xml = tmp_path / "g.xml"
    xml.write_text('<Document><Minutia X="10,5" Y="12" Angle="0.3"/></Document>')   # comma decimal
    pts = ft.read_gt(str(xml))
    assert pts.shape == (1, 2) and np.isclose(pts[0, 0], 10.5) and np.isclose(pts[0, 1], 12.0)
    h = ft.heatmap(np.array([[10.0, 12.0]], np.float32), 32, 32, 3.0)
    assert h.shape == (32, 32) and h.min() >= 0.0 and h.max() <= 1.0
    assert h[12, 10] > 0.9                                # peak at (y=12, x=10)
