"""Scratch: run the NumPy MLP on the project's folds (fold files in outputs/oof/<name>/).

Usage: python mlp_run.py --name m_a --model '{"hidden": [512, 256], ...}' [--bins 16]
The slurm array task id t runs fold seed 42 + t // 5, fold t % 5 (or --seed/--fold).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pgonzale/ML_p1")
sys.path.insert(0, "/scratch/pgonzale/p1x")
import mlp  # noqa: E402
from p1 import cleaning  # noqa: E402
from p1 import models as M  # noqa: E402
from p1 import transforms as T  # noqa: E402
from p1.config import load_config  # noqa: E402
from p1.data import load_clean  # noqa: E402
from p1.evaluation import kfold_indices, run_fold  # noqa: E402

T.TRANSFORMS["mlp_encoding"] = mlp.MLPEncoding
M.MODELS["mlp"] = mlp.MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="{}")
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument(
        "--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    args = ap.parse_args()
    seed, fold = 42 + args.task // 5, args.task % 5
    start = time.perf_counter()
    ds = load_clean()
    numeric = [j for j, f in enumerate(ds.feature_names) if f in cleaning.NUMERIC]
    config = load_config("gbdt_tuned")
    config["name"] = args.name
    config["seed"] = seed
    config["transforms"] = [
        {"name": "mlp_encoding", "numeric": numeric, "n_bins": args.bins}
    ]
    config["model"] = {"name": "mlp", **json.loads(args.model)}
    splits = kfold_indices(
        len(ds.y_train), config["k_folds"], seed, config["subsample"]
    )
    scores = run_fold(config, ds, splits, fold)
    print(
        f"{args.name} seed {seed} fold {fold}: F1 {scores['f1']:.4f} AUC {scores['auc']:.4f}"
        f" ({time.perf_counter() - start:.0f} s)"
    )


if __name__ == "__main__":
    main()
