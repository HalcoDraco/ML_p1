from typing import Any

Config = dict[str, Any]

DEFAULTS: Config = {
    "seed": 42,
    "k_folds": 5,
    "subsample": 1.0,  # fraction of training rows used by `cv`
    "transforms": [],
}


def load_config(preset: str) -> Config:
    """Read a preset and fill its missing keys from `DEFAULTS`.

    Args:
        preset: Preset name, as in `configs/<preset>.toml`, or path to a `.toml` file.

    Returns:
        The preset, with `name` set to the file name if missing.
    """
    pass


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Return a copy of a preset with `key=value` overrides applied.

    Values are read as TOML (`0.1`, `[1, 2]`) and otherwise kept as strings.
    Keys are top-level (`seed`), `model.<option>` or `transforms.<name>.<option>`.

    Args:
        config: Preset to change.
        overrides: Items such as `model.lambda_=0.1`.

    Returns:
        The changed preset.
    """
    pass


def expand_grid(config: Config) -> list[Config]:
    """Turn the `[grid]` table of a preset into one preset per combination.

    Args:
        config: Preset, with or without a `grid` table.

    Returns:
        The variants, named like `ridge_poly[degree=2,lambda_=0.001]`, or
        `[config]` if there is no grid.
    """
    pass
