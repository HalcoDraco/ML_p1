import time

import numpy as np

from p1.config import Config
from p1.data import Dataset
from p1.metrics import accuracy, f1_score, roc_auc
from p1.pipeline import build_pipeline


def kfold_indices(n: int, k: int, seed: int, fraction: float = 1.0) -> list[np.ndarray]:
    """Shuffle `range(n)` with `seed` and split it into `k` folds of indices.

    Only the first `fraction` of the shuffled indices is split, so that a
    subsample of the rows gives quicker runs.
    """
    rows = np.random.default_rng(seed).permutation(n)[: round(fraction * n)]
    return np.array_split(rows, k)


def cross_validate(
    config: Config, ds: Dataset, verbose: bool = False
) -> dict[str, float]:
    """Estimate a preset's scores by k-fold cross-validation on the training data.

    A new pipeline is fitted on each training split, so its transforms and its
    decision threshold never see the validation fold. With the same `seed`,
    every preset uses the same folds.

    Args:
        config: Preset to evaluate. Uses its `k_folds`, `seed` and `subsample`.
        ds: Clean dataset.
        verbose: Print the F1 score of each fold as it is computed.

    Returns:
        The mean and standard deviation over the folds of the F1 score
        (`f1_mean`, `f1_std`), the accuracy (`acc_mean`, `acc_std`) and the ROC
        AUC (`auc_mean`, `auc_std`); the mean fitted threshold
        (`threshold_mean`); and the run time in `seconds`.
    """
    start = time.perf_counter()
    n = len(ds.y_train)
    folds = kfold_indices(n, config["k_folds"], config["seed"], config["subsample"])
    scores = {"f1": [], "acc": [], "auc": [], "threshold": []}
    for i, valid in enumerate(folds):
        train = np.concatenate(folds[:i] + folds[i + 1 :])
        pipeline = build_pipeline(config, ds.feature_names)
        pipeline.fit(ds.x_train[train], ds.y_train[train])
        y_valid = ds.y_train[valid]
        score = pipeline.predict_score(ds.x_train[valid])
        y_pred = (score >= pipeline.model.threshold_).astype(int)
        scores["f1"].append(f1_score(y_valid, y_pred))
        scores["acc"].append(accuracy(y_valid, y_pred))
        scores["auc"].append(roc_auc(score, y_valid))
        scores["threshold"].append(pipeline.model.threshold_)
        if verbose:
            seconds = time.perf_counter() - start
            fold_f1 = scores["f1"][-1]
            print(
                f"  fold {i + 1}/{len(folds)}: F1 {fold_f1:.4f} ({seconds:.0f} s)",
                flush=True,
            )
    return {
        "f1_mean": float(np.mean(scores["f1"])),
        "f1_std": float(np.std(scores["f1"])),
        "acc_mean": float(np.mean(scores["acc"])),
        "acc_std": float(np.std(scores["acc"])),
        "auc_mean": float(np.mean(scores["auc"])),
        "auc_std": float(np.std(scores["auc"])),
        "threshold_mean": float(np.mean(scores["threshold"])),
        "seconds": time.perf_counter() - start,
    }
