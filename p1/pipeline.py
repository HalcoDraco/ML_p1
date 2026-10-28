from typing import Any, Self

import numpy as np

from p1 import cleaning
from p1.config import Config, load_config
from p1.models import MODELS, Model
from p1.transforms import TRANSFORMS, Transform


class Pipeline:
    """Transforms followed by a model, fitted and applied as one unit.

    Args:
        transforms: Steps applied in order before the model.
        model: Predictor trained on the transformed features (`models.MODELS`).
        seed: Seed for NumPy's global random generator, set at the start of `fit`.
    """

    def __init__(self, transforms: list[Transform], model: Model, seed: int) -> None:
        self.transforms = transforms
        self.model = model
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Fit each transform in order on the training data, then the model.

        Training-only row operations, such as undersampling, also go here.

        Args:
            x: Training features, shape (N, D).
            y: Training labels in {0, 1}, shape (N,).

        Returns:
            The fitted pipeline.
        """
        np.random.seed(self.seed)  # the SGD of implementations.py uses this generator
        for step in self.transforms:
            x = step.fit_transform(x, y)
        self.model.fit(x, y)
        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted transforms and return the model's scores."""
        return self.model.predict_score(self.transform(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted transforms and return labels in {0, 1}."""
        return self.model.predict(self.transform(x))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted transforms in order."""
        for step in self.transforms:
            x = step.transform(x)
        return x


def build_pipeline(config: Config, feature_names: list[str] | None = None) -> Pipeline:
    """Create an unfitted pipeline from a preset.

    A transform given as a string uses no options; a table gives its `name` and
    its options. A `columns` option, a column group or a list of feature names
    (see `column_indices`), is turned into column indices, so steps with one
    must come before any step that moves columns, such as `onehot`. So must a
    `categorical` model option (GBDT), which is resolved the same way.

    Args:
        config: Preset with `transforms`, `model` and `seed`.
        feature_names: Column names, so transform options can refer to columns.

    Returns:
        The pipeline described by the preset.

    Raises:
        ValueError: If a transform or model name is not registered, or a step
            picks columns by name after a step that moved them.
    """
    steps = []
    moved_by = None  # name of the first step that removed or reordered columns
    for item in config["transforms"]:
        name, options = _name_and_options(item, TRANSFORMS, "transform")
        if "columns" in options:
            if moved_by:
                raise ValueError(
                    f"transform '{name}' picks columns by name, "
                    f"so it must come before '{moved_by}'"
                )
            options["columns"] = column_indices(options["columns"], feature_names)
        step = TRANSFORMS[name](**options)
        if step.moves_columns and not moved_by:
            moved_by = name
        steps.append(step)
    name, options = _name_and_options(config["model"], MODELS, "model")
    if "categorical" in options:
        if moved_by:
            raise ValueError(
                f"model option 'categorical' picks columns by name, "
                f"so no transform may move them, as '{moved_by}' does"
            )
        options["categorical"] = column_indices(options["categorical"], feature_names)
    if "init" in options:  # GBDT boosted from the scores of another preset
        options["init"] = build_pipeline(load_config(options["init"]), feature_names)
    return Pipeline(steps, MODELS[name](**options), config["seed"])


def column_indices(
    columns: str | list[str], feature_names: list[str] | None
) -> list[int]:
    """Return the indices, in feature order, of a column group or list of names.

    The groups are "numeric" (`cleaning.NUMERIC`), "heavy_tailed"
    (`cleaning.HEAVY_TAILED`) and "categorical" (every feature not numeric).

    Raises:
        ValueError: If the names are missing, or a group or name is unknown.
    """
    if feature_names is None:
        raise ValueError("a `columns` option needs the feature names")
    if columns == "numeric":
        names = cleaning.NUMERIC
    elif columns == "heavy_tailed":
        names = cleaning.HEAVY_TAILED
    elif columns == "categorical":
        names = [name for name in feature_names if name not in cleaning.NUMERIC]
    elif isinstance(columns, str):
        raise ValueError(
            f"unknown column group '{columns}': use numeric, categorical, "
            "heavy_tailed or a list of feature names"
        )
    else:
        names = columns
    unknown = sorted(set(names) - set(feature_names))
    if unknown:
        raise ValueError(f"unknown feature names: {', '.join(unknown)}")
    wanted = set(names)
    return [j for j, name in enumerate(feature_names) if name in wanted]


def _name_and_options(
    item: str | dict[str, Any], registry: dict[str, type], kind: str
) -> tuple[str, dict[str, Any]]:
    """Split a preset item into its registered name and its options."""
    options = {"name": item} if isinstance(item, str) else dict(item)
    name = options.pop("name")
    if name not in registry:
        raise ValueError(f"unknown {kind} '{name}', use one of: {', '.join(registry)}")
    return name, options
