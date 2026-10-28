import copy
import itertools
import tomllib
from pathlib import Path
from typing import Any

from p1.paths import CONFIGS_DIR

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
    path = Path(preset) if preset.endswith(".toml") else CONFIGS_DIR / f"{preset}.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)
    return {"name": path.stem, **copy.deepcopy(DEFAULTS), **config}


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
    config = copy.deepcopy(config)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' is not of the form KEY=VALUE")
        _set(config, key.strip(), _parse(raw.strip()))
    return config


def expand_grid(config: Config) -> list[Config]:
    """Turn the `[grid]` table of a preset into one preset per combination.

    Args:
        config: Preset, with or without a `grid` table.

    Returns:
        The variants, named like `ridge_poly[degree=2,lambda_=0.001]`, or
        `[config]` if there is no grid.
    """
    grid = config.get("grid")
    if not grid:
        return [config]
    variants = []
    for values in itertools.product(*grid.values()):
        variant = copy.deepcopy(config)
        del variant["grid"]
        for key, value in zip(grid, values):
            _set(variant, key, value)
        labels = [f"{key.split('.')[-1]}={value}" for key, value in zip(grid, values)]
        variant["name"] = f"{config['name']}[{','.join(labels)}]"
        variants.append(variant)
    return variants


def _parse(raw: str) -> Any:
    """Read a value written in TOML (`0.1`, `true`, `[1, 2]`), or keep the string."""
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def _set(config: Config, key: str, value: Any) -> None:
    """Set the option `key` of a preset, in place.

    A transform given by its name only is turned into a table first. If the
    preset has the same transform twice, the first one is changed.
    """
    parts = key.split(".")
    if len(parts) == 1:
        config[key] = value
    elif len(parts) == 2 and parts[0] == "model":
        config["model"][parts[1]] = value
    elif len(parts) == 3 and parts[0] == "transforms":
        _, name, option = parts
        steps = config["transforms"]
        for i, step in enumerate(steps):
            if step == name or (isinstance(step, dict) and step["name"] == name):
                if isinstance(step, str):
                    steps[i] = step = {"name": step}
                step[option] = value
                return
        raise ValueError(f"cannot set '{key}': the preset has no transform '{name}'")
    else:
        raise ValueError(
            f"cannot set '{key}': keys are KEY, model.KEY or transforms.NAME.KEY"
        )
