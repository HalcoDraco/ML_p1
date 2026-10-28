import json
import time
from typing import Any

import numpy as np

from p1 import ensemble
from p1.config import Config
from p1.data import Dataset, drop_features
from p1.metrics import best_threshold, f1_score, log_loss, roc_auc
from p1.pipeline import build_pipeline


def kfold_indices(n: int, k: int, seed: int, fraction: float = 1.0) -> list[np.ndarray]:
    """Shuffle `range(n)` with `seed` and split it into `k` folds of indices.

    Only the first `fraction` of the shuffled indices is split, so that a
    subsample of the rows gives quicker runs.
    """
    rows = np.random.default_rng(seed).permutation(n)[: round(fraction * n)]
    return np.array_split(rows, k)


def cross_validate(
    config: Config,
    ds: Dataset,
    verbose: bool = False,
    folds: list[int] | None = None,
) -> dict[str, Any] | None:
    """Estimate a preset's scores by k-fold cross-validation on the training data.

    A new pipeline is fitted on each training split, so its transforms and its
    decision threshold never see the validation fold. With the same `seed`,
    every preset uses the same folds. Each fold's scores are saved to a fold
    file (`run_fold`), so the folds can also run as separate jobs.

    Args:
        config: Preset to evaluate. Uses its `k_folds`, `seed`, `subsample`
            and `drop`.
        ds: Clean dataset.
        verbose: Print the F1 score of each fold as it is computed.
        folds: Indices of the folds to run; all of them if None.

    Returns:
        `ensemble.summarize` of the run once all its folds are saved, else None.
    """
    start = time.perf_counter()
    ds = drop_features(ds, config["drop"])
    n = len(ds.y_train)
    splits = kfold_indices(n, config["k_folds"], config["seed"], config["subsample"])
    for i in range(len(splits)) if folds is None else folds:
        scores = run_fold(config, ds, splits, i)
        if verbose:
            seconds = time.perf_counter() - start
            print(
                f"  fold {i + 1}/{len(splits)}: F1 {scores['f1']:.4f}, "
                f"AUC {scores['auc']:.4f} ({seconds:.0f} s)",
                flush=True,
            )
    try:
        return ensemble.summarize(config["name"], config["seed"])
    except FileNotFoundError:
        return None


def run_fold(
    config: Config, ds: Dataset, splits: list[np.ndarray], i: int
) -> dict[str, float]:
    """Fit a preset on all folds but fold `i` and save the fold file.

    The file (`ensemble.fold_path`) holds the validation rows, their labels and
    scores, the scores of the test rows, the model's own threshold, the preset
    and the run time. For models with `staged_scores` (GBDT), it also holds
    the validation AUC, log-loss and best-threshold F1 after every block of
    trees, which shows the best number of trees.

    Returns:
        The fold's F1 with the model's own threshold (`f1`) and its AUC (`auc`).
    """
    start = time.perf_counter()
    valid = splits[i]
    train = np.concatenate(splits[:i] + splits[i + 1 :])
    pipeline = build_pipeline(config, ds.feature_names)
    pipeline.fit(ds.x_train[train], ds.y_train[train])
    x_valid, y_valid = ds.x_train[valid], ds.y_train[valid]
    score = pipeline.predict_score(x_valid)
    threshold = pipeline.model.threshold_
    fields = {
        "valid_idx": valid,
        "valid_y": y_valid.astype(np.int8),
        "valid_score": score,
        "test_score": pipeline.predict_score(ds.x_test),
        "threshold": threshold,
        "config": json.dumps(config),
    }
    if hasattr(pipeline.model, "staged_scores"):
        stages = pipeline.model.staged_scores(pipeline.transform(x_valid))
        fields |= _staged_metrics(stages, y_valid)
    fields["seconds"] = time.perf_counter() - start
    ensemble.save_fold(ensemble.fold_path(config["name"], config["seed"], i), fields)
    return {
        "f1": f1_score(y_valid, (score >= threshold).astype(int)),
        "auc": roc_auc(score, y_valid),
    }


def _staged_metrics(
    stages: list[tuple[int, np.ndarray]], y: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the AUC, log-loss and best-threshold F1 of each stage's scores."""
    metrics = {"stage_trees": [], "stage_auc": [], "stage_logloss": []}
    metrics["stage_f1"] = []
    for n_trees, score in stages:
        metrics["stage_trees"].append(n_trees)
        metrics["stage_auc"].append(roc_auc(score, y))
        metrics["stage_logloss"].append(log_loss(score, y))
        y_pred = (score >= best_threshold(score, y)).astype(int)
        metrics["stage_f1"].append(f1_score(y, y_pred))
    return {key: np.array(values) for key, values in metrics.items()}
