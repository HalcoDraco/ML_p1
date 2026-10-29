"""Out-of-fold (OOF) scores: fold files, CV summaries, paired comparisons, blends.

`evaluation.run_fold` saves one file per run, fold seed and fold, in
`outputs/oof/<run>/seed<seed>_fold<k>.npz`. It holds the scores of the fold's
model on its validation rows and on the test set. Everything here works from
these files only, so nothing is refitted: a blend of finished runs takes
seconds.

Scores are probabilities (logistic regression, GBDT). Blends average them on
the logit scale and threshold the result.
"""

import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from p1.metrics import accuracy, best_threshold, f1_score, roc_auc
from p1.paths import OOF_DIR

# Two-sided 95% critical values of Student's t, by degrees of freedom.
T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 9: 2.262, 14: 2.145}

Fold = dict[str, Any]


def run_dir(name: str) -> Path:
    """Return the folder of a run's fold files, with a name made safe for paths."""
    return OOF_DIR / re.sub(r"[^\w.=+-]+", "_", name).strip("_")


def fold_path(name: str, seed: int, fold: int) -> Path:
    """Return the fold file of a run, fold seed and fold index."""
    return run_dir(name) / f"seed{seed}_fold{fold}.npz"


def save_fold(path: Path, fields: Fold) -> None:
    """Write a fold file.

    Args:
        path: File from `fold_path`.
        fields: At least `valid_idx`, `valid_y`, `valid_score`, `test_score`,
            `threshold` (the fold model's own) and `config` (JSON text).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fields)


def seeds_of(name: str) -> list[int]:
    """Return the fold seeds that have at least one fold file for a run."""
    found = {re.match(r"seed(\d+)_", p.name) for p in run_dir(name).glob("*.npz")}
    return sorted(int(match[1]) for match in found if match)


def load_run(name: str, seed: int) -> list[Fold]:
    """Load the fold files of a run for one fold seed, in fold order.

    Raises:
        FileNotFoundError: If a fold is missing.
        ValueError: If the folds were run with different presets.
    """
    paths = run_dir(name).glob(f"seed{seed}_fold*.npz")
    folds = {}
    for path in paths:
        with np.load(path) as f:
            fold = {key: f[key] for key in f.files}
        fold["config"] = str(fold["config"])
        folds[int(re.search(r"fold(\d+)", path.name)[1])] = fold
    if not folds:
        raise FileNotFoundError(f"no fold files for {name}, seed {seed}")
    k = json.loads(next(iter(folds.values()))["config"])["k_folds"]
    missing = sorted(set(range(k)) - set(folds))
    if missing:
        raise FileNotFoundError(f"{name}, seed {seed}: folds {missing} are missing")
    if len({fold["config"] for fold in folds.values()}) > 1:
        raise ValueError(f"{name}, seed {seed}: folds come from different presets")
    return [folds[i] for i in range(k)]


def nested_thresholds(folds: list[Fold]) -> list[float]:
    """Return, for each fold, the F1-optimal threshold of the other folds' scores.

    Applied to its own fold, it gives an F1 estimate in which the threshold
    never saw the rows it is scored on.
    """
    thresholds = []
    for i in range(len(folds)):
        others = folds[:i] + folds[i + 1 :]
        score = np.concatenate([fold["valid_score"] for fold in others])
        y = np.concatenate([fold["valid_y"] for fold in others])
        thresholds.append(best_threshold(score, y))
    return thresholds


def summarize(name: str, seed: int) -> dict[str, Any]:
    """Return the cross-validation scores of a run for one fold seed.

    Returns:
        Means and standard deviations over the folds of the F1 with the fold
        model's own threshold (`f1_mean`, `f1_std`), of the F1 with the nested
        threshold (`f1_nested_mean`, `f1_nested_std`; `f1_folds` lists it per
        fold), of the accuracy and of the ROC AUC; the F1 of the pooled OOF
        scores at their best threshold (`f1_pooled`, slightly optimistic) and
        that threshold (`threshold_pooled`); the mean own threshold; and the
        summed fit time of the folds in `seconds`.
    """
    folds = load_run(name, seed)
    own, nested, acc, auc = [], [], [], []
    for fold, threshold in zip(folds, nested_thresholds(folds)):
        y, score = fold["valid_y"], fold["valid_score"]
        own.append(f1_score(y, (score >= fold["threshold"]).astype(int)))
        nested.append(f1_score(y, (score >= threshold).astype(int)))
        acc.append(accuracy(y, (score >= threshold).astype(int)))
        auc.append(roc_auc(score, y))
    score = np.concatenate([fold["valid_score"] for fold in folds])
    y = np.concatenate([fold["valid_y"] for fold in folds])
    threshold = best_threshold(score, y)
    return {
        "f1_mean": float(np.mean(own)),
        "f1_std": float(np.std(own)),
        "f1_nested_mean": float(np.mean(nested)),
        "f1_nested_std": float(np.std(nested)),
        "f1_pooled": f1_score(y, (score >= threshold).astype(int)),
        "acc_mean": float(np.mean(acc)),
        "acc_std": float(np.std(acc)),
        "auc_mean": float(np.mean(auc)),
        "auc_std": float(np.std(auc)),
        "threshold_mean": float(np.mean([fold["threshold"] for fold in folds])),
        "threshold_pooled": threshold,
        "seconds": float(sum(fold.get("seconds", 0.0) for fold in folds)),
        "f1_folds": [round(f1, 5) for f1 in nested],
    }


def compare(name_a: str, name_b: str) -> dict[str, float]:
    """Compare two runs on the folds they share, with the corrected t-test.

    The per-fold differences of the nested F1 are paired over every (fold seed,
    fold) that both runs have. The Nadeau-Bengio correction inflates their
    variance by n_valid / n_train, since the training sets overlap.

    Returns:
        The mean difference `diff` (A - B), its standard deviation `std`, the
        number of pairs `pairs`, the corrected `t` and the bounds `low`, `high`
        of its 95% interval.
    """
    diffs, k = [], None
    for seed in sorted(set(seeds_of(name_a)) & set(seeds_of(name_b))):
        f1_a = summarize(name_a, seed)["f1_folds"]
        f1_b = summarize(name_b, seed)["f1_folds"]
        diffs += [a - b for a, b in zip(f1_a, f1_b)]
        k = len(f1_a)
    if len(diffs) < 2:
        raise ValueError(f"{name_a} and {name_b} share fewer than 2 folds")
    d = np.array(diffs)
    std = float(d.std(ddof=1))
    scale = np.sqrt((1 / len(d) + 1 / (k - 1)) * std**2)
    df = len(d) - 1
    critical = T_CRITICAL[max(key for key in T_CRITICAL if key <= df)]
    return {
        "diff": float(d.mean()),
        "std": std,
        "pairs": len(d),
        "t": float(d.mean() / scale) if scale > 0 else float("inf"),
        "low": float(d.mean() - critical * scale),
        "high": float(d.mean() + critical * scale),
    }


def logit(p: np.ndarray) -> np.ndarray:
    """Return log(p / (1 - p)), with p clipped away from 0 and 1."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Return 1 / (1 + exp(-z))."""
    return 0.5 * (1 + np.tanh(0.5 * z))


def blend(
    names: list[str], out: str, weights: list[float] | None = None
) -> tuple[list[float], dict[int, dict[str, Any]]]:
    """Blend runs by a weighted mean of their logits, and save it as a run.

    The runs must share fold seeds, and so their folds. Without `weights`,
    the weights are searched on a grid with step 0.05 (two runs) or 0.1 (more)
    for the best mean pooled OOF F1 over the shared seeds. Each blended fold
    file gets the nested threshold, so `summarize` and `compare` treat the
    blend like any other run.

    Args:
        names: Runs to blend.
        out: Name of the blended run.
        weights: One weight per run, normalized to sum 1; searched if None.

    Returns:
        The weights and the `summarize` result of each shared seed.
    """
    seeds = sorted(set.intersection(*(set(seeds_of(name)) for name in names)))
    if not seeds:
        raise ValueError(f"the runs {names} share no fold seed")
    runs = {seed: [load_run(name, seed) for name in names] for seed in seeds}
    for seed, members in runs.items():
        for folds in members[1:]:
            for fold, first in zip(folds, members[0]):
                if not np.array_equal(fold["valid_idx"], first["valid_idx"]):
                    raise ValueError(f"seed {seed}: the runs use different folds")
    pooled = {  # (runs, rows) logits and labels of the pooled OOF rows
        seed: (
            np.array(
                [np.concatenate([logit(f["valid_score"]) for f in m]) for m in members]
            ),
            np.concatenate([fold["valid_y"] for fold in members[0]]),
        )
        for seed, members in runs.items()
    }
    if weights is None:
        weights = _search_weights(pooled, len(names))
    w = np.array(weights, dtype=float) / np.sum(weights)
    summaries = {}
    for seed, members in runs.items():
        k = len(members[0])
        config = {"name": out, "k_folds": k, "seed": seed, "members": names}
        config["weights"] = w.tolist()
        folds = []
        for i in range(k):
            fold = members[0][i]
            valid = sum(wm * logit(m[i]["valid_score"]) for wm, m in zip(w, members))
            test = sum(wm * logit(m[i]["test_score"]) for wm, m in zip(w, members))
            folds.append(
                {
                    "valid_idx": fold["valid_idx"],
                    "valid_y": fold["valid_y"],
                    "valid_score": sigmoid(valid),
                    "test_score": sigmoid(test),
                    "config": json.dumps(config),
                }
            )
        for i, threshold in enumerate(nested_thresholds(folds)):
            folds[i]["threshold"] = threshold
            save_fold(fold_path(out, seed, i), folds[i])
        summaries[seed] = summarize(out, seed)
    return w.tolist(), summaries


def _search_weights(
    pooled: dict[int, tuple[np.ndarray, np.ndarray]], n_runs: int
) -> list[float]:
    """Return the grid weights with the best mean pooled OOF F1 over seeds."""
    steps = 20 if n_runs == 2 else 10
    best, best_f1 = None, -1.0
    for head in itertools.product(range(steps + 1), repeat=n_runs - 1):
        if sum(head) > steps:
            continue
        w = np.array([*head, steps - sum(head)]) / steps
        f1 = 0.0
        for z, y in pooled.values():
            score = w @ z
            f1 += f1_score(y, (score >= best_threshold(score, y)).astype(int))
        if f1 > best_f1:
            best, best_f1 = w.tolist(), f1
    return best


def test_scores(name: str) -> tuple[np.ndarray, float, float]:
    """Return a run's test scores, decision threshold and OOF positive rate.

    The test score is the sigmoid of the mean logit of all its fold models
    (every fold of every fold seed): an average of models fitted on different
    80% of the rows. The threshold is the F1-optimal one on the OOF scores of
    all seeds pooled.

    Returns:
        The test scores, the threshold and the fraction of OOF rows it flags.
    """
    scores, valid, y = [], [], []
    for seed in seeds_of(name):
        for fold in load_run(name, seed):
            scores.append(logit(fold["test_score"]))
            valid.append(fold["valid_score"])
            y.append(fold["valid_y"])
    valid, y = np.concatenate(valid), np.concatenate(y)
    threshold = best_threshold(valid, y)
    return (
        sigmoid(np.mean(scores, axis=0)),
        threshold,
        float(np.mean(valid >= threshold)),
    )
