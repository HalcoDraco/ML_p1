from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def missing_fraction(x: np.ndarray) -> np.ndarray:
    """Return the fraction of NaN values in each column of `x`."""
    pass


def plot_by_class(values: np.ndarray, y: np.ndarray, title: str) -> Figure:
    """Plot the distribution of one feature separately for each class.

    Args:
        values: One feature column, shape (N,).
        y: Labels in {0, 1}, shape (N,).
        title: Figure title, usually the feature name.

    Returns:
        The figure, ready for `save_figure`.
    """
    pass


def save_figure(fig: Figure, name: str) -> Path:
    """Save `fig` as `report/figures/<name>.pdf` and return the file path."""
    pass
