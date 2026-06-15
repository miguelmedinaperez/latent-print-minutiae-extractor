"""Dump PyFing/LEADER's Keras weights + reference I/O for a PyTorch port (runs in pyfing-tf2).

Writes:
  data/crops/leader_weights.npz   - every layer's weights, keys "<layer>::<i>"
  data/crops/leader_layers.json   - ordered layer list with type + minimal config
  data/crops/leader_parity.npz    - a real 320x320 input patch (raw 0-255) and the Keras
                                    outputs: full 5ch `out`, and the pre-concat heads
                                    pos(1) / dir2(2, pre-arctan2) / typ(1) — for parity testing.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import glob, json
import numpy as np
import cv2 as cv
import keras
from pyfing.minutiae import Leader

WORK = "."  # run inside a checkout with PyFing installed
ext = Leader()
model = ext.model

# 1. weights
wd = {}
for lyr in model.layers:
    for i, w in enumerate(lyr.get_weights()):
        wd[f"{lyr.name}::{i}"] = np.asarray(w)
np.savez(f"{WORK}/data/crops/leader_weights.npz", **wd)

# 2. layer order + config
KEYS = ("filters", "kernel_size", "dilation_rate", "strides", "padding", "epsilon",
        "activation", "use_bias", "axis", "depth_multiplier", "size", "pool_size", "rate")
meta = []
for lyr in model.layers:
    cfg = lyr.get_config()
    meta.append({"name": lyr.name, "type": lyr.__class__.__name__,
                 "config": {k: cfg[k] for k in KEYS if k in cfg},
                 "nweights": len(lyr.get_weights())})
with open(f"{WORK}/data/crops/leader_layers.json", "w") as f:
    json.dump(meta, f, indent=1, default=str)

# 3. parity I/O — a real 320x320 patch fed raw (0-255), like run()
img = cv.imread(sorted(glob.glob("*.bmp") or glob.glob("**/*.bmp", recursive=True))[0], cv.IMREAD_GRAYSCALE)
patch = img[:320, :320].astype(np.float32)
if patch.shape != (320, 320):                       # pad if the first image is small
    patch = cv.copyMakeBorder(patch, 0, 320 - patch.shape[0], 0, 320 - patch.shape[1],
                              cv.BORDER_CONSTANT, value=float(img[0, 0]))
x = patch[np.newaxis, ..., np.newaxis]
out = model(x, training=False).numpy()              # (1,320,320,5)
sub = keras.Model(model.input, [model.get_layer(n).output for n in
                                ("head_conv_pos", "head_conv_dir", "head_conv_typ")])
pos, dir2, typ = [t.numpy() for t in sub(x, training=False)]
np.savez(f"{WORK}/data/crops/leader_parity.npz",
         x=x.astype(np.float32), out=out, pos=pos, dir2=dir2, typ=typ)

print("layers:", len(meta), "| weight tensors:", len(wd))
print("parity shapes:", "x", x.shape, "out", out.shape, "pos", pos.shape, "dir2", dir2.shape)
print("out channel ranges:", [(round(float(out[..., c].min()), 3), round(float(out[..., c].max()), 3)) for c in range(5)])
