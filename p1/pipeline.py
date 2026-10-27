from typing import Self

import numpy as np

from p1.config import Config
from p1.models import LinearModel
from p1.transforms import Transform


class Pipeline:
    """Transforms followed by a model, fitted and applied as one unit.

    Args:
        transforms: Steps applied in order before the model.
        model: Predictor trained on the transformed features.
        seed: Seed for NumPy's global random generator, set at the start of `fit`.
    """

    def __init__(
        self, transforms: list[Transform], model: LinearModel, seed: int
    ) -> None:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Fit each transform in order on the training data, then the model.

        Training-only row operations, such as undersampling, also go here.

        Args:
            x: Training features, shape (N, D).
            y: Training labels in {0, 1}, shape (N,).

        Returns:
            The fitted pipeline.
        """
        pass

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted transforms and return the model's scores."""
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted transforms and return labels in {0, 1}."""
        pass


def build_pipeline(config: Config, feature_names: list[str] | None = None) -> Pipeline:
    """Create an unfitted pipeline from a preset.

    A transform given as a string uses no options; a table gives its `name` and
    its options.

    Args:
        config: Preset with `transforms`, `model` and `seed`.
        feature_names: Column names, so transform options can refer to columns.

    Returns:
        The pipeline described by the preset.

    Raises:
        ValueError: If a transform or model name is not registered.
    """
    pass
