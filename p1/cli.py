import argparse
import csv
import json
import pickle
from datetime import datetime
from typing import Any

import numpy as np

from p1 import ensemble
from p1.config import Config, apply_overrides, expand_grid, load_config
from p1.data import drop_features, load_clean, write_submission
from p1.evaluation import cross_validate
from p1.paths import CV_RESULTS, MODELS_DIR, OOF_DIR, SUBMISSIONS_DIR
from p1.pipeline import build_pipeline

CV_COLUMNS = [
    "timestamp",
    "name",
    "seed",
    "f1_nested_mean",
    "f1_nested_std",
    "f1_mean",
    "f1_std",
    "f1_pooled",
    "acc_mean",
    "acc_std",
    "auc_mean",
    "auc_std",
    "threshold_mean",
    "threshold_pooled",
    "k_folds",
    "subsample",
    "seconds",
    "f1_folds",
    "config",
]


def run_cv(
    presets: list[str],
    overrides: list[str],
    folds: list[int] | None = None,
    variant: int | None = None,
) -> None:
    """Cross-validate presets and their grids, then rank them.

    Prints a table sorted by F1 and appends one row per variant to
    `outputs/cv_results.csv`. Every fold also writes its fold file
    (`ensemble.fold_path`), which `collect`, `compare` and `blend` use.

    Args:
        presets: Preset names or `.toml` paths.
        overrides: `key=value` items applied to every preset.
        folds: Run only these folds, as one slurm task does; `collect` then
            summarizes the run once all its folds exist.
        variant: Run only this grid variant, in `expand_grid` order.
    """
    ds = load_clean()
    results = []
    for preset in presets:
        config = apply_overrides(load_config(preset), overrides)
        variants = expand_grid(config)
        if variant is not None:
            variants = [variants[variant]]
        for config in variants:
            print(f"{config['name']}:", flush=True)  # shows progress in a log file too
            scores = cross_validate(config, ds, verbose=True, folds=folds)
            if folds is None:
                _append_cv_rows([(config, scores)])  # saved at once, in case of a crash
                results.append((config, scores))
    if results:
        _print_ranking(results)


def collect(patterns: list[str]) -> None:
    """Summarize runs from their fold files, rank them and log them.

    Args:
        patterns: Run names or shell patterns, such as `gbdt*`, matched against
            the folders of `outputs/oof/`.
    """
    runs = sorted({path.name for pattern in patterns for path in OOF_DIR.glob(pattern)})
    results = []
    for name in runs:
        for seed in ensemble.seeds_of(name):
            try:
                scores = ensemble.summarize(name, seed)
            except FileNotFoundError as error:
                print(f"skipped: {error}")
                continue
            config = json.loads(ensemble.load_run(name, seed)[0]["config"])
            results.append(({**config, "name": name}, scores))
    _append_cv_rows(results)
    _print_ranking(results)


def compare(name_a: str, name_b: str) -> None:
    """Print the paired difference of two runs' nested F1 and its corrected t."""
    r = ensemble.compare(name_a, name_b)
    print(
        f"{name_a} - {name_b}: {r['diff']:+.4f} (sd {r['std']:.4f}, "
        f"{r['pairs']} paired folds), corrected t = {r['t']:.2f}, "
        f"95% interval [{r['low']:+.4f}, {r['high']:+.4f}]"
    )


def blend(names: list[str], out: str, weights: list[float] | None) -> None:
    """Blend runs on the logit scale, save the blend as run `out` and show it."""
    w, summaries = ensemble.blend(names, out, weights)
    print(f"{out}: weights " + ", ".join(f"{n} {x:.2f}" for n, x in zip(names, w)))
    config = {"name": out, "members": names, "weights": w}
    rows = [({**config, "seed": seed}, s) for seed, s in summaries.items()]
    _append_cv_rows(rows)
    _print_ranking(rows)


def show_stages(names: list[str], seed: int) -> None:
    """Print each GBDT run's fold-averaged validation metrics by number of trees.

    The rows with the best log-loss and the best F1 are marked.
    """
    for name in names:
        s = ensemble.stages(name, seed)
        best_loss, best_f1 = np.argmin(s["logloss"]), np.argmax(s["f1"])
        print(f"{name} (seed {seed}):\n  trees     AUC  logloss      F1")
        for i, trees in enumerate(s["trees"]):
            mark = (" <- log-loss" if i == best_loss else "") + (
                " <- F1" if i == best_f1 else ""
            )
            print(
                f"  {trees:5d}  {s['auc'][i]:.4f}  {s['logloss'][i]:.5f}"
                f"  {s['f1'][i]:.4f}{mark}"
            )


def submit(names: list[str]) -> None:
    """Write `outputs/submissions/<run>.csv` from each run's fold files.

    The test scores are averaged over the run's fold models and thresholded
    at the F1-optimal threshold of its pooled OOF scores (`ensemble.test_scores`).
    """
    ds = load_clean()
    answered = ~np.isnan(ds.x_test[:, ds.feature_names.index("HAREHAB1")])
    for name in names:
        score, threshold, oof_rate = ensemble.test_scores(name)
        y_pred = (score >= threshold).astype(int)
        path = SUBMISSIONS_DIR / f"{ensemble.run_dir(name).name}.csv"
        write_submission(ds.test_ids, y_pred, path)
        print(
            f"{name}: wrote {path} (threshold {threshold:.4f}; "
            f"{y_pred.mean():.2%} of test rows predicted positive, "
            f"{oof_rate:.2%} of OOF rows; {y_pred[answered].sum()} of the "
            f"{answered.sum()} test rows that answered HAREHAB1 predicted 1)"
        )


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
        kept = drop_features(ds, config["drop"])
        pipeline = build_pipeline(config, kept.feature_names)
        pipeline.fit(kept.x_train, kept.y_train)
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
        config = json.loads((MODELS_DIR / f"{name}.json").read_text())
        y_pred = pipeline.predict(drop_features(ds, config.get("drop", [])).x_test)
        path = SUBMISSIONS_DIR / f"{name}.csv"
        write_submission(ds.test_ids, y_pred, path)
        print(f"{name}: wrote {path} ({y_pred.mean():.1%} predicted positive)")


def _append_cv_rows(results: list[tuple[Config, dict[str, Any]]]) -> None:
    """Append cross-validation results to `outputs/cv_results.csv`.

    A file written with other columns is first rewritten with the current
    ones, the missing values left empty.
    """
    CV_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    old_rows = []
    if CV_RESULTS.exists():
        with open(CV_RESULTS, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CV_COLUMNS:
                old_rows = list(reader)
    mode = "a" if CV_RESULTS.exists() and not old_rows else "w"
    with open(CV_RESULTS, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CV_COLUMNS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            writer.writerows(old_rows)
        for config, scores in results:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "name": config["name"],
                "seed": config.get("seed"),
                "k_folds": config.get("k_folds"),
                "subsample": config.get("subsample"),
                "config": json.dumps(config),
            }
            for key, value in scores.items():
                row[key] = json.dumps(value) if isinstance(value, list) else value
            writer.writerow(
                {k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}
            )


def _print_ranking(results: list[tuple[Config, dict[str, Any]]]) -> None:
    """Print cross-validation results as a table sorted by the nested F1.

    The nested F1 uses, for each fold, the threshold that is best on the other
    folds' OOF scores; "own" uses each fold model's own threshold.
    """
    if not results:
        print("no finished run")
        return
    results = sorted(results, key=lambda r: r[1]["f1_nested_mean"], reverse=True)
    width = max(len("run"), *(len(config["name"]) for config, _ in results))
    print(
        f"\n{'run':<{width}}  {'seed':>4}  {'F1 nested':^15}  {'own':>6}"
        f"  {'pooled':>6}  {'AUC':>6}  {'threshold':>9}  {'seconds':>7}"
    )
    for config, s in results:
        print(
            f"{config['name']:<{width}}  {config.get('seed', ''):>4}"
            f"  {s['f1_nested_mean']:.4f} ± {s['f1_nested_std']:.4f}"
            f"  {s['f1_mean']:.4f}  {s['f1_pooled']:.4f}  {s['auc_mean']:.4f}"
            f"  {s['threshold_pooled']:>9.3f}  {s['seconds']:>7.0f}"
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
    cv_parser.add_argument(
        "--fold",
        dest="folds",
        type=int,
        action="append",
        help="run only this fold, from 0 (repeatable); `collect` summarizes later",
    )
    cv_parser.add_argument(
        "--variant", type=int, help="run only this grid variant, from 0"
    )
    predict_parser = commands.add_parser("predict", help="write submission CSVs")
    predict_parser.add_argument("names", nargs="+", help="names of trained presets")
    collect_parser = commands.add_parser(
        "collect", help="summarize runs from their fold files"
    )
    collect_parser.add_argument("runs", nargs="+", help="run names or patterns")
    compare_parser = commands.add_parser(
        "compare", help="paired comparison of two runs (corrected t-test)"
    )
    compare_parser.add_argument("run_a")
    compare_parser.add_argument("run_b")
    blend_parser = commands.add_parser("blend", help="blend runs' OOF scores")
    blend_parser.add_argument("runs", nargs="+", help="runs to blend")
    blend_parser.add_argument("--name", required=True, help="name of the blend")
    blend_parser.add_argument(
        "--weights", type=float, nargs="+", help="one per run; searched if omitted"
    )
    stages_parser = commands.add_parser(
        "stages", help="GBDT validation metrics by number of trees"
    )
    stages_parser.add_argument("runs", nargs="+", help="run names")
    stages_parser.add_argument("--seed", type=int, default=42, help="fold seed")
    submit_parser = commands.add_parser(
        "submit", help="write submission CSVs from runs' fold files"
    )
    submit_parser.add_argument("runs", nargs="+", help="run names")

    args = parser.parse_args(argv)
    if args.command == "cv":
        run_cv(args.presets, args.overrides, args.folds, args.variant)
    elif args.command == "train":
        train(args.presets, args.overrides)
    elif args.command == "predict":
        predict(args.names)
    elif args.command == "collect":
        collect(args.runs)
    elif args.command == "compare":
        compare(args.run_a, args.run_b)
    elif args.command == "blend":
        blend(args.runs, args.name, args.weights)
    elif args.command == "stages":
        show_stages(args.runs, args.seed)
    else:
        submit(args.runs)


if __name__ == "__main__":
    main()
