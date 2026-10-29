"""Scratch: summarize runs (best stage by fold-mean log-loss, nested F1), per fold seed.

Usage: python report.py PATTERN [PATTERN ...] [--seed 42]
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1 import ensemble  # noqa: E402
from p1.paths import OOF_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--seed", type=int, action="append")
    args = ap.parse_args()
    names = sorted({p.name for pat in args.patterns for p in OOF_DIR.glob(pat)})
    print(
        f"{'run':34s} seed  best-T  logloss     AUC   F1stg | nested    AUC   pooled thr"
    )
    for name in names:
        for seed in args.seed or ensemble.seeds_of(name):
            try:
                folds = ensemble.load_run(name, seed)
                s = ensemble.summarize(name, seed)
            except (FileNotFoundError, ValueError) as e:
                print(f"{name:34s} {seed}  incomplete: {str(e)[:60]}")
                continue
            if "stage_trees" in folds[0]:
                n = min(len(f["stage_trees"]) for f in folds)
                ll = np.mean([f["stage_logloss"][:n] for f in folds], axis=0)
                auc = np.mean([f["stage_auc"][:n] for f in folds], axis=0)
                f1 = np.mean([f["stage_f1"][:n] for f in folds], axis=0)
                b = int(np.argmin(ll))
                stage = f"{folds[0]['stage_trees'][b]:6d}  {ll[b]:.5f}  {auc[b]:.4f}  {f1[b]:.4f}"
            else:
                stage = " " * 31
            print(
                f"{name:34s} {seed}  {stage} | {s['f1_nested_mean']:.4f}  "
                f"{s['auc_mean']:.4f}  {s['f1_pooled']:.4f} {s['threshold_pooled']:.3f}"
            )


if __name__ == "__main__":
    main()
