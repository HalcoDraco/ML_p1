"""Scratch: feature-engineering variants of the tuned GBDT on the project's folds.

Registers extra transforms at runtime and runs `p1.evaluation.run_fold`, so
the fold files land in outputs/oof/<name>/ like any `p1 cv` run.

Usage: python fe.py --name x_tord --variant tord [--set model.n_trees=600 ...]
The slurm array task id t runs fold seed 42 + t // 5, fold t % 5.
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p1 import cleaning  # noqa: E402
from p1 import transforms as T  # noqa: E402
from p1.config import apply_overrides, load_config  # noqa: E402
from p1.data import load_clean  # noqa: E402
from p1.evaluation import kfold_indices, run_fold  # noqa: E402


class RowCounts(T.Transform):
    """Append one column per group: count of `code` (or of NaN) in the group's columns."""

    def __init__(self, groups):
        self.groups = groups

    def transform(self, x):
        extra = []
        for idx, code in self.groups:
            sub = x[:, idx]
            if code == "nan":
                extra.append(np.isnan(sub).sum(axis=1))
            elif code == "sum":
                extra.append(
                    np.where(np.isnan(sub).all(axis=1), np.nan, np.nansum(sub, axis=1))
                )
            else:
                extra.append((sub == code).sum(axis=1))
        return np.hstack([x, np.column_stack(extra).astype(x.dtype)])


class TargetOrder(T.Transform):
    """Replace each level of a column by its smoothed training positive rate.

    NaN stays NaN (the GBDT learns its side); unseen levels get the prior.
    """

    def __init__(self, columns=None, smooth=50.0):
        self.columns = columns
        self.smooth = smooth

    def fit(self, x, y=None):
        self.columns_ = T._indices(self.columns, x)
        y = np.asarray(y, dtype=float)
        self.prior_ = prior = float(y.mean())
        self.maps_ = []
        for j in self.columns_:
            col = x[:, j]
            ok = ~np.isnan(col)
            values, inv = np.unique(col[ok], return_inverse=True)
            n = np.bincount(inv, minlength=len(values))
            pos = np.bincount(inv, weights=y[ok], minlength=len(values))
            self.maps_.append((values, (pos + self.smooth * prior) / (n + self.smooth)))
        return self

    def transform(self, x):
        x = x.copy()
        for j, (values, rate) in zip(self.columns_, self.maps_):
            col = x[:, j]
            ok = ~np.isnan(col)
            v = col[ok]
            pos = np.minimum(np.searchsorted(values, v), len(values) - 1)
            x[ok, j] = np.where(values[pos] == v, rate[pos], self.prior_)
        return x


class LRFeature(T.Transform):
    """Append the logit of a logistic-regression preset (stacking).

    Training rows get out-of-fold scores from an inner k-fold, so the GBDT
    sees on them what it will see on new rows; other rows get the scores of
    the preset refitted on all training rows.
    """

    def __init__(self, feature_names, preset="step2/logistic", inner_folds=5, seed=0):
        self.feature_names = feature_names
        self.preset = preset
        self.inner_folds = inner_folds
        self.seed = seed

    def _pipeline(self):
        from p1.pipeline import build_pipeline

        return build_pipeline(load_config(self.preset), self.feature_names)

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def fit_transform(self, x, y=None):
        n = len(y)
        parts = np.array_split(
            np.random.default_rng(self.seed).permutation(n), self.inner_folds
        )
        oof = np.empty(n)
        for k in range(self.inner_folds):
            rest = np.concatenate(parts[:k] + parts[k + 1 :])
            oof[parts[k]] = (
                self._pipeline().fit(x[rest], y[rest]).predict_score(x[parts[k]])
            )
            print(f"    inner LR fold {k + 1}/{self.inner_folds} done", flush=True)
        self.full_ = self._pipeline().fit(x, y)
        return np.column_stack([x, self._logit(oof)])

    def fit(self, x, y=None):
        self.fit_transform(x, y)
        return self

    def transform(self, x):
        return np.column_stack([x, self._logit(self.full_.predict_score(x))])


T.TRANSFORMS["row_counts"] = RowCounts
T.TRANSFORMS["target_order"] = TargetOrder
T.TRANSFORMS["lr_feature"] = LRFeature

CHRONIC = "BPHIGH4 TOLDHI2 CVDSTRK3 ASTHMA3 CHCSCNCR CHCOCNCR CHCCOPD1 HAVARTH3 ADDEPEV2 CHCKIDNY DIABETE3"
DISABILITY = "QLACTLM2 USEEQUIP BLIND DECIDE DIFFWALK DIFFDRES DIFFALON"


RISK = "_RFHYPE5 _RFCHOL _RFSMOK3 _RFBMI5 _RFHLTH _TOTINDA"


def count_groups(ds, tokens=("nan", "yn", "med")):
    names = ds.feature_names
    index = {n: i for i, n in enumerate(names)}
    yesno = []
    for i, n in enumerate(names):
        if n in cleaning.NUMERIC or n.startswith("_") or n == "SEX":
            continue
        col = ds.x_train[:, i]
        values = set(np.unique(col[~np.isnan(col)]).tolist())
        if values <= {1.0, 2.0, 7.0, 9.0} and {1.0, 2.0} <= values:
            yesno.append(i)
    chronic = [index[n] for n in CHRONIC.split()]
    disability = [index[n] for n in DISABILITY.split()]
    groups = []
    if "nan" in tokens:
        groups.append([list(range(len(names))), "nan"])
    if "yn" in tokens:
        groups += [[yesno, 1.0], [yesno, 2.0], [yesno, 7.0], [yesno, 9.0]]
    if "med" in tokens:
        groups += [[chronic, 1.0], [chronic, 7.0], [disability, 1.0]]
    if "risk" in tokens:
        groups.append([[index[n] for n in RISK.split()], 2.0])
    if "care" in tokens:
        care = [index[n] for n in "CHECKUP1 CHOLCHK FLUSHOT6 PNEUVAC3 BPMEDS".split()]
        groups.append([care, 1.0])
    if "days" in tokens:
        groups.append([[index["PHYSHLTH"], index["MENTHLTH"]], "sum"])
    return groups


def variant_transforms(variant, ds, smooth):
    steps = []
    tokens = variant.split("_")
    if "cnt" in tokens:
        tokens += ["nan", "yn", "med"]
    if "lrf" in tokens:
        steps.append({"name": "lr_feature", "feature_names": ds.feature_names})
    groups = count_groups(ds, tokens)
    if groups:
        steps.append({"name": "row_counts", "groups": groups})
    if "tord" in tokens:
        steps.append(
            {"name": "target_order", "columns": "categorical", "smooth": smooth}
        )
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--variant", required=True, help="contains cnt and/or tord, or base"
    )
    ap.add_argument("--preset", default="gbdt_tuned")
    ap.add_argument("--smooth", type=float, default=50.0)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument(
        "--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    args = ap.parse_args()
    seed, fold = 42 + args.task // 5, args.task % 5

    ds = load_clean()
    config = apply_overrides(load_config(args.preset), args.overrides)
    config["name"] = args.name
    config["seed"] = seed
    config["transforms"] = variant_transforms(args.variant, ds, args.smooth) + list(
        config["transforms"]
    )
    splits = kfold_indices(
        len(ds.y_train), config["k_folds"], seed, config["subsample"]
    )
    scores = run_fold(config, ds, splits, fold)
    print(
        f"{args.name} seed {seed} fold {fold}: F1 {scores['f1']:.4f} AUC {scores['auc']:.4f}"
    )


if __name__ == "__main__":
    main()
