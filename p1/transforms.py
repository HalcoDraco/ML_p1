from typing import Self

import numpy as np


class Transform:
    """Preprocessing step fitted on training data and applied at inference.

    Options come from the preset as keyword arguments. State learned in `fit`
    is stored in attributes ending with `_`.
    """

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Learn the step's state from training data. Does nothing by default.

        Args:
            x: Training features, shape (N, D).
            y: Training labels in {0, 1}, shape (N,). Most steps ignore them.

        Returns:
            The fitted step.
        """
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted step to features of shape (N, D)."""
        pass

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit on `x`, then transform it."""
        pass


class ImputeMedian(Transform):
    """Replace NaNs with the training median of their column."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Store the median of each column."""
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Fill NaNs with the stored medians."""
        pass


class Standardize(Transform):
    """Scale each column to zero mean and unit variance."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Store the mean and standard deviation of each column."""
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Center and scale with the stored statistics."""
        pass


class Whiten(Transform):
    """Decorrelate the columns and scale them to unit variance (PCA whitening)."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Store the mean and the whitening matrix of the training data."""
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Center and multiply by the stored whitening matrix."""
        pass


class Polynomial(Transform):
    """Append the powers 2 to `degree` of each column, without cross terms.

    Args:
        degree: Highest power added.
    """

    def __init__(self, degree: int) -> None:
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return `x` with the power columns appended."""
        pass


TRANSFORMS: dict[str, type[Transform]] = {
    "impute_median": ImputeMedian,
    "standardize": Standardize,
    "whiten": Whiten,
    "poly": Polynomial,
}
