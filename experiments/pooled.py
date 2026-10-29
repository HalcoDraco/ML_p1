"""Scratch: pooled OOF AUC / log-loss / nested F1 per run (mean over its fold seeds).

Usage: python pooled.py RUN [RUN ...]
"""

import sys

import numpy as np

sys.path.insert(0, "/home/pgonzale/ML_p1")
from p1 import ensemble  # noqa: E402
from p1.metrics import log_loss, roc_auc  # noqa: E402

print(f"{'run':24s} seeds  pooled AUC  pooled logloss  nested F1   (per seed)")
for name in sys.argv[1:]:
    rows = []
    for seed in ensemble.seeds_of(name):
        try:
            folds = ensemble.load_run(name, seed)
        except (FileNotFoundError, ValueError):
            continue
        p = np.concatenate([f["valid_score"] for f in folds])
        y = np.concatenate([f["valid_y"] for f in folds])
        rows.append(
            (
                roc_auc(p, y),
                log_loss(p, y),
                ensemble.summarize(name, seed)["f1_nested_mean"],
            )
        )
    if not rows:
        print(f"{name:24s} no complete seed")
        continue
    a = np.array(rows)
    print(
        f"{name:24s} {len(a):5d}  {a[:, 0].mean():.5f}     {a[:, 1].mean():.5f}       "
        f"{a[:, 2].mean():.4f}   {np.round(a[:, 2], 4)}"
    )
