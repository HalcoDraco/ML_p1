import numpy as np


def clean(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Apply the fixed cleaning rules to a feature matrix.

    The rules come from the BRFSS codebook and learn nothing from the data, so
    train and test are cleaned the same way. Steps that learn from the data are
    transforms in `p1/transforms.py`.

    Args:
        x: Raw features, shape (N, D).
        feature_names: Name of each column of `x`.

    Returns:
        The clean features and the names of their columns.
    """
    pass


def build_clean_dataset() -> None:
    """Clean the raw train and test data and save it to `data/processed/clean.npz`."""
    pass
