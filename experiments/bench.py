"""Scratch benchmark: external GBDT libraries on the project's exact folds.

NOT part of the repo and never used for a submission (the assignment bans
external ML libraries). It only measures how far our NumPy GBDT is from
LightGBM / XGBoost / CatBoost on the same folds.

Writes fold files outputs/oof/<name>_t<T>/seed<seed>_fold<k>.npz in the
p1.ensemble format, so `p1 collect / compare / blend / stages` work on them.

Usage (one fold per slurm array task):
  python bench.py --model lgb --name bench_lgb_mimic --params '{...}' --trees 1500 \
      --save 550,1000,1500 --seed 42 --fold 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pgonzale/ML_p1")
from p1 import cleaning, ensemble  # noqa: E402
from p1.data import load_clean  # noqa: E402
from p1.evaluation import kfold_indices  # noqa: E402
from p1.metrics import best_threshold, f1_score, log_loss, roc_auc  # noqa: E402


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def staged_metrics(stages, y):
    out = {"stage_trees": [], "stage_auc": [], "stage_logloss": [], "stage_f1": []}
    for t, p in stages:
        out["stage_trees"].append(t)
        out["stage_auc"].append(roc_auc(p, y))
        out["stage_logloss"].append(log_loss(p, y))
        out["stage_f1"].append(f1_score(y, (p >= best_threshold(p, y)).astype(int)))
    return {k: np.array(v) for k, v in out.items()}


def run_lgb(params, xtr, ytr, cat_idx, n_trees):
    import lightgbm as lgb

    params = {"objective": "binary", "verbose": -1, **params}
    dtr = lgb.Dataset(
        xtr,
        ytr,
        categorical_feature=cat_idx if cat_idx else "auto",
        params={k: v for k, v in params.items() if k in ("max_bin", "min_data_in_bin")},
        free_raw_data=False,
    )
    booster = lgb.train(params, dtr, num_boost_round=n_trees)
    return lambda x, t: booster.predict(x, num_iteration=t, raw_score=True)


def run_xgb(params, xtr, ytr, cat_idx, n_trees):
    import xgboost as xgb

    params = {"objective": "binary:logistic", "tree_method": "hist", **params}
    dtr = xgb.QuantileDMatrix(xtr, ytr, max_bin=params.get("max_bin", 256))
    booster = xgb.train(params, dtr, num_boost_round=n_trees)

    def predict(x, t):
        return booster.predict(
            xgb.DMatrix(x), iteration_range=(0, t), output_margin=True
        )

    return predict


def to_frame(x, cat_idx):
    import pandas as pd

    df = pd.DataFrame(x, columns=[f"f{j}" for j in range(x.shape[1])])
    for j in cat_idx:
        col = x[:, j]
        df[f"f{j}"] = np.where(np.isnan(col), -1, col).astype(np.int64)
    return df


def run_cat(params, xtr, ytr, cat_idx, n_trees):
    from catboost import CatBoostClassifier, Pool

    model = CatBoostClassifier(
        iterations=n_trees, loss_function="Logloss", verbose=250, **params
    )
    model.fit(Pool(to_frame(xtr, cat_idx), ytr, cat_features=cat_idx or None))

    def predict(x, t):
        pool = Pool(to_frame(x, cat_idx), cat_features=cat_idx or None)
        return model.predict(pool, prediction_type="RawFormulaVal", ntree_end=t)

    return predict


RUNNERS = {"lgb": run_lgb, "xgb": run_xgb, "cat": run_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=RUNNERS, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--trees", type=int, required=True)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument(
        "--save", required=True, help="tree counts to save, comma-separated"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--fold", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    ap.add_argument(
        "--cat", action="store_true", help="declare the categorical features"
    )
    args = ap.parse_args()
    params = json.loads(args.params)

    start = time.perf_counter()
    ds = load_clean()
    cat_idx = (
        [j for j, f in enumerate(ds.feature_names) if f not in cleaning.NUMERIC]
        if args.cat
        else []
    )
    splits = kfold_indices(len(ds.y_train), 5, args.seed)
    valid = splits[args.fold]
    train = np.concatenate(splits[: args.fold] + splits[args.fold + 1 :])
    xtr, ytr = ds.x_train[train], ds.y_train[train].astype(float)
    xva, yva = ds.x_train[valid], ds.y_train[valid]
    predict = RUNNERS[args.model](params, xtr, ytr, cat_idx, args.trees)
    fit_seconds = time.perf_counter() - start
    print(f"fitted in {fit_seconds:.0f} s", flush=True)

    trees = sorted(set(range(args.every, args.trees + 1, args.every)) | {args.trees})
    stages = [(t, sigmoid(predict(xva, t))) for t in trees]
    metrics = staged_metrics(stages, yva)
    best = int(np.argmin(metrics["stage_logloss"]))
    print(
        f"best stage {trees[best]} trees: logloss {metrics['stage_logloss'][best]:.5f}"
        f" AUC {metrics['stage_auc'][best]:.4f} F1 {metrics['stage_f1'][best]:.4f}",
        flush=True,
    )
    config = {
        "name": args.name,
        "k_folds": 5,
        "seed": args.seed,
        "model": args.model,
        "params": params,
        "cat": args.cat,
        "trees": args.trees,
    }
    for t in [int(s) for s in args.save.split(",")]:
        p_valid = dict(stages)[t]
        p_train = sigmoid(predict(xtr, t))
        fields = {
            "valid_idx": valid,
            "valid_y": yva.astype(np.int8),
            "valid_score": p_valid,
            "test_score": sigmoid(predict(ds.x_test, t)),
            "threshold": best_threshold(p_train, ytr),
            "config": json.dumps({**config, "saved_trees": t}),
            "seconds": fit_seconds,
            **metrics,
        }
        name = f"{args.name}_t{t}"
        ensemble.save_fold(ensemble.fold_path(name, args.seed, args.fold), fields)
        print(f"saved {name} fold {args.fold}", flush=True)
    print(f"done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
