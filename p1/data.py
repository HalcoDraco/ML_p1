import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import helpers
from p1.paths import RAW_CACHE, RAW_DIR, RAW_ZIP

RAW_CSVS = ("x_train.csv", "y_train.csv", "x_test.csv")


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
    if not RAW_CACHE.exists():
        _extract_raw_csvs()
        x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(
            str(RAW_DIR)
        )
        with open(RAW_DIR / "x_train.csv") as f:
            feature_names = f.readline().strip().split(",")[1:]  # drop "Id"
        RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            RAW_CACHE,
            x_train=x_train,
            y_train=(y_train == 1).astype(int),  # {-1, 1} -> {0, 1}
            x_test=x_test,
            train_ids=train_ids,
            test_ids=test_ids,
            feature_names=np.array(feature_names),
        )
    with np.load(RAW_CACHE) as f:
        return Dataset(
            x_train=f["x_train"],
            y_train=f["y_train"],
            x_test=f["x_test"],
            train_ids=f["train_ids"],
            test_ids=f["test_ids"],
            feature_names=f["feature_names"].tolist(),
        )


def _extract_raw_csvs() -> None:
    """Extract the CSVs missing from `data/raw/` out of the dataset zip.

    The zip stores them in a `dataset/` folder; they are written directly in
    `data/raw/`, where `helpers.load_csv_data` expects them.
    """
    if all((RAW_DIR / name).exists() for name in RAW_CSVS):
        return
    with zipfile.ZipFile(RAW_ZIP) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name.endswith(".csv") and not (RAW_DIR / name).exists():
                with archive.open(member) as src, open(RAW_DIR / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)


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
