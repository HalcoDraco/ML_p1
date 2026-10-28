import argparse
import csv
import json
import pickle
from datetime import datetime

from p1.config import Config, apply_overrides, expand_grid, load_config
from p1.data import load_clean, write_submission
from p1.evaluation import cross_validate
from p1.paths import CV_RESULTS, MODELS_DIR, SUBMISSIONS_DIR
from p1.pipeline import build_pipeline

CV_COLUMNS = [
    "timestamp",
    "name",
    "f1_mean",
    "f1_std",
    "acc_mean",
    "acc_std",
    "auc_mean",
    "auc_std",
    "threshold_mean",
    "k_folds",
    "subsample",
    "seconds",
    "config",
]


def run_cv(presets: list[str], overrides: list[str]) -> None:
    """Cross-validate presets and their grids, then rank them.

    Prints a table sorted by F1 and appends one row per variant to
    `outputs/cv_results.csv`.

    Args:
        presets: Preset names or `.toml` paths.
        overrides: `key=value` items applied to every preset.
    """
    ds = load_clean()
    results = []
    for preset in presets:
        config = apply_overrides(load_config(preset), overrides)
        for variant in expand_grid(config):
            print(f"{variant['name']}:", flush=True)  # shows progress in a log file too
            scores = cross_validate(variant, ds, verbose=True)
            _append_cv_row(variant, scores)  # saved at once, in case a later run fails
            results.append((variant, scores))
    _print_ranking(results)


def train(presets: list[str], overrides: list[str]) -> None:
    """Fit each preset on all training data and save it to `outputs/models/`.

    The `[grid]` table is ignored: the preset's base values are used.

    Args:
        presets: Preset names or `.toml` paths.
        overrides: `key=value` items applied to every preset.
    """
    ds = load_clean()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for preset in presets:
        config = apply_overrides(load_config(preset), overrides)
        if "grid" in config:
            print(f"{config['name']}: [grid] ignored, training with the base values")
        pipeline = build_pipeline(config, ds.feature_names)
        pipeline.fit(ds.x_train, ds.y_train)
        path = MODELS_DIR / f"{config['name']}.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        path.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")
        print(
            f"{config['name']}: saved {path} (threshold {pipeline.model.threshold_:.3f})"
        )


def predict(names: list[str]) -> None:
    """Write `outputs/submissions/<name>.csv` for each trained preset.

    Args:
        names: Names of presets saved by `train`.
    """
    ds = load_clean()
    for name in names:
        with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
            pipeline = pickle.load(f)
        y_pred = pipeline.predict(ds.x_test)
        path = SUBMISSIONS_DIR / f"{name}.csv"
        write_submission(ds.test_ids, y_pred, path)
        print(f"{name}: wrote {path} ({y_pred.mean():.1%} predicted positive)")


def _append_cv_row(config: Config, scores: dict[str, float]) -> None:
    """Append one cross-validation result to `outputs/cv_results.csv`."""
    CV_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CV_RESULTS.exists()
    with open(CV_RESULTS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "name": config["name"],
                **{key: round(value, 5) for key, value in scores.items()},
                "k_folds": config["k_folds"],
                "subsample": config["subsample"],
                "config": json.dumps(config),
            }
        )


def _print_ranking(results: list[tuple[Config, dict[str, float]]]) -> None:
    """Print the cross-validation results as a table sorted by F1."""
    results = sorted(results, key=lambda result: result[1]["f1_mean"], reverse=True)
    width = max(len("preset"), *(len(config["name"]) for config, _ in results))
    print(
        f"\n{'preset':<{width}}  {'F1':^15}  {'accuracy':>8}  {'AUC':>6}"
        f"  {'threshold':>9}  {'seconds':>7}"
    )
    for config, s in results:
        print(
            f"{config['name']:<{width}}  {s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
            f"  {s['acc_mean']:>8.4f}  {s['auc_mean']:>6.4f}"
            f"  {s['threshold_mean']:>9.3f}  {s['seconds']:>7.0f}"
        )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the chosen command."""
    parser = argparse.ArgumentParser(
        prog="p1", description="Part 3: run presets of transforms + model."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    cv_parser = commands.add_parser("cv", help="cross-validate presets")
    train_parser = commands.add_parser("train", help="fit and save presets")
    for subparser in (cv_parser, train_parser):
        subparser.add_argument("presets", nargs="+", help="preset names or .toml paths")
        subparser.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a preset value, e.g. model.lambda_=0.1 (repeatable)",
        )
    predict_parser = commands.add_parser("predict", help="write submission CSVs")
    predict_parser.add_argument("names", nargs="+", help="names of trained presets")

    args = parser.parse_args(argv)
    if args.command == "cv":
        run_cv(args.presets, args.overrides)
    elif args.command == "train":
        train(args.presets, args.overrides)
    else:
        predict(args.names)
