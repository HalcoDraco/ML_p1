"""Reproduce the best submission from the raw data.

Rebuilds the clean dataset, fits each member preset of `configs/best.toml` on
all the training rows, averages the members' test logits with the preset's
weights, and writes `outputs/submissions/best.csv` with the preset's frozen
threshold. The weights and the threshold were chosen on out-of-fold scores
(`p1 blend`), never on the test set. The members are retrained instead of
loaded, so nothing in `outputs/` is needed beforehand.

Usage: `uv run python run.py [--jobs N]`, N members fitted in parallel
(default 4; the logistic regression needs about 13 GB, a GBDT about 3 GB).
"""

import argparse
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from p1.cleaning import build_clean_dataset
from p1.config import apply_overrides, load_config
from p1.data import drop_features, load_clean, write_submission
from p1.ensemble import logit, sigmoid
from p1.paths import CONFIGS_DIR, SUBMISSIONS_DIR
from p1.pipeline import build_pipeline


def fit_member(member: dict[str, Any]) -> np.ndarray:
    """Fit one member on all the training rows and return its test scores.

    Args:
        member: `preset` (name or path) and optional `set`, a list of
            `key=value` overrides as on the command line.
    """
    start = time.perf_counter()
    config = apply_overrides(load_config(member["preset"]), member.get("set", []))
    ds = drop_features(load_clean(), config["drop"])
    pipeline = build_pipeline(config, ds.feature_names)
    pipeline.fit(ds.x_train, ds.y_train)
    seconds = time.perf_counter() - start
    print(f"fitted {member['preset']} {member.get('set', [])} in {seconds:.0f} s")
    return pipeline.predict_score(ds.x_test)


def main() -> None:
    """Fit the members, blend them and write `outputs/submissions/best.csv`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4, help="members fitted at once")
    args = parser.parse_args()
    build_clean_dataset()
    with open(CONFIGS_DIR / "best.toml", "rb") as f:
        best = tomllib.load(f)
    with ProcessPoolExecutor(args.jobs) as pool:
        scores = list(pool.map(fit_member, best["members"]))
    weights = np.array(best["weights"]) / np.sum(best["weights"])
    score = sigmoid(sum(w * logit(s) for w, s in zip(weights, scores)))
    y_pred = (score >= best["threshold"]).astype(int)
    ds = load_clean()
    path = SUBMISSIONS_DIR / "best.csv"
    write_submission(ds.test_ids, y_pred, path)
    answered = ~np.isnan(ds.x_test[:, ds.feature_names.index("HAREHAB1")])
    print(
        f"wrote {path}: {y_pred.mean():.2%} of the test rows predicted positive; "
        f"{y_pred[answered].sum()} of the {answered.sum()} that answered "
        "HAREHAB1 predicted 1"
    )


if __name__ == "__main__":
    main()
