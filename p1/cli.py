import argparse


def run_cv(presets: list[str], overrides: list[str]) -> None:
    """Cross-validate presets and their grids, then rank them.

    Prints a table sorted by F1 and appends one row per variant to
    `outputs/cv_results.csv`.

    Args:
        presets: Preset names or `.toml` paths.
        overrides: `key=value` items applied to every preset.
    """
    pass


def train(presets: list[str], overrides: list[str]) -> None:
    """Fit each preset on all training data and save it to `outputs/models/`.

    The `[grid]` table is ignored: the preset's base values are used.

    Args:
        presets: Preset names or `.toml` paths.
        overrides: `key=value` items applied to every preset.
    """
    pass


def predict(names: list[str]) -> None:
    """Write `outputs/submissions/<name>.csv` for each trained preset.

    Args:
        names: Names of presets saved by `train`.
    """
    pass


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
