"""Tests for leader.viz. The first test PINS the angle-painting convention (the direction line is
drawn at -angle in image pixel coordinates, y down) so a regression there fails loudly; the second
covers the CLI overlay path end-to-end."""
import sys
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conftest import inject_one_peak_forward
import leader.viz as viz
from leader import MinutiaeExtractor


def test_draw_uses_negated_angle_in_pixel_coords():
    img = np.zeros((200, 168), np.uint8)
    m = {"x": 80, "y": 100, "angle": 0.6, "quality": 0.9}
    fig, ax = plt.subplots()
    viz.draw(ax, img, [m])
    lines = [ln for ln in ax.lines if len(ln.get_xdata()) == 2]   # the 2-point direction segment
    assert lines, "no direction line was drawn"
    xd, yd = lines[-1].get_xdata(), lines[-1].get_ydata()
    L = max(16.0, 0.018 * max(img.shape))
    a = -m["angle"]                                               # the documented painting convention
    assert np.isclose(xd[0], 80) and np.isclose(yd[0], 100)       # line starts at the minutia
    assert np.isclose(xd[1], 80 + L * np.cos(a))                  # dx = cos(-angle)
    assert np.isclose(yd[1], 100 + L * np.sin(a))                 # dy = sin(-angle), y increases downward
    plt.close(fig)


def test_viz_main_writes_overlay(tmp_path, monkeypatch, synthetic_print):
    ex = MinutiaeExtractor(device="cpu", half=False)
    inject_one_peak_forward(ex)
    monkeypatch.setattr(viz, "MinutiaeExtractor", lambda *a, **k: ex)
    img = tmp_path / "p.png"; cv.imwrite(str(img), synthetic_print)
    out = tmp_path / "overlay.png"
    monkeypatch.setattr(sys, "argv", ["viz", str(img), "--out", str(out)])
    viz.main()
    assert out.exists() and out.stat().st_size > 0
