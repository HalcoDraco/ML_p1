import numpy as np

from p1.config import Config
from p1.data import Dataset


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the F1 score of labels in {0, 1}, with 1 as the positive class."""
    pass


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of correct predictions."""
    pass


def kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    """Shuffle `range(n)` with `seed` and split it into `k` folds of indices."""
    pass


def cross_validate(config: Config, ds: Dataset) -> dict[str, float]:
    """Estimate a preset's scores by k-fold cross-validation on the training data.

    A new pipeline is fitted on each training split, so its transforms never see
    the validation fold. With the same `seed`, every preset uses the same folds.

    Args:
        config: Preset to evaluate. Uses its `k_folds`, `seed` and `subsample`.
        ds: Clean dataset.

    Returns:
        `f1_mean`, `f1_std`, `acc_mean`, `acc_std` and `seconds`.
    """
    pass
