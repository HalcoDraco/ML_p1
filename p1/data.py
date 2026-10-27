from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    """Train and test data, raw or clean.

    Features have one row per respondent and NaN for missing answers. Labels are
    in {0, 1}; they become {-1, 1} only in submission files.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    feature_names: list[str]


def load_raw() -> Dataset:
    """Load the raw CSVs, extracting them from the zip if they are missing.

    Parsing the CSVs is slow, so the arrays are cached in `data/processed/raw.npz`.
    """
    pass


def load_clean() -> Dataset:
    """Load the clean dataset written by `p1.cleaning.build_clean_dataset`.

    Warns if `p1/cleaning.py` changed after the clean dataset was written.

    Raises:
        FileNotFoundError: If the clean dataset has not been built yet.
    """
    pass


def write_submission(ids: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Write a submission CSV, mapping labels from {0, 1} to {-1, 1}.

    Args:
        ids: Test ids, shape (N,).
        y_pred: Predicted labels in {0, 1}, shape (N,).
        path: CSV file to write.
    """
    pass
