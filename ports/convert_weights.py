"""Convert the verified Keras weight-dumps into clean torch state_dict checkpoints
for the redistributable minutiae_superset package."""
import sys
from pathlib import Path
import torch

WORK = "."  # set to your working dir (reference conversion code)
sys.path.insert(0, WORK)

from minutiae_superset._layers import WSrc                      # noqa: E402
from minutiae_superset.fingernet import FingerNet               # noqa: E402
from minutiae_superset.coarsenet import CoarseNet               # noqa: E402
from minutiae_superset.finenet import FineNet, Loader           # noqa: E402

crops = f"{WORK}/data/crops"
out = Path(f"{WORK}/minutiae_superset/weights")
out.mkdir(parents=True, exist_ok=True)

fn = FingerNet(WSrc(f"{crops}/fingernet_keras_weights.pkl"))
torch.save(fn.state_dict(), out / "fingernet.pt")
print("fingernet.pt:", sum(p.numel() for p in fn.parameters()), "params")

cn = CoarseNet(WSrc(f"{crops}/coarsenet_keras_weights.pkl"))
torch.save(cn.state_dict(), out / "coarsenet.pt")
print("coarsenet.pt:", sum(p.numel() for p in cn.parameters()), "params")

ld = Loader(f"{crops}/finenet_keras_weights.pkl")
fe = FineNet(ld)
unused = set(ld.d) - ld.used
assert not unused, f"FineNet unused Keras layers: {sorted(unused)[:6]}"
torch.save(fe.state_dict(), out / "finenet.pt")
print("finenet.pt:", sum(p.numel() for p in fe.parameters()), "params; all Keras weights consumed")

for f in sorted(out.glob("*.pt")):
    print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")
