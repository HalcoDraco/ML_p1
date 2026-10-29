"""Scratch: paired per-fold staged log-loss / AUC differences of runs vs a reference run, at given tree counts."""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1 import ensemble

ref = sys.argv[1]
tree_ref = int(sys.argv[2])
for arg in sys.argv[3:]:
    name, t = arg.split("@")
    t = int(t)
    dl, da = [], []
    for seed in (42, 43, 44):
        try:
            a = ensemble.load_run(name, seed)
            b = ensemble.load_run(ref, seed)
        except Exception:
            continue
        ia = list(a[0]["stage_trees"]).index(t)
        ib = list(b[0]["stage_trees"]).index(tree_ref)
        dl += [
            fa["stage_logloss"][ia] - fb["stage_logloss"][ib] for fa, fb in zip(a, b)
        ]
        da += [fa["stage_auc"][ia] - fb["stage_auc"][ib] for fa, fb in zip(a, b)]
    dl, da = np.array(dl), np.array(da)
    if len(dl) == 0:
        print(f"{arg:28s} no complete seed")
        continue
    print(
        f"{arg:28s} folds {len(dl):2d}  dlogloss {dl.mean():+.5f} (neg {np.mean(dl < 0):.0%})  dAUC {da.mean():+.5f} (pos {np.mean(da > 0):.0%})"
    )
