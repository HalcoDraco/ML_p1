from typing import Self

import numpy as np


class Transform:
    """Preprocessing step fitted on training data and applied at inference.

    Options come from the preset as keyword arguments. State learned in `fit`
    is stored in attributes ending with `_`. A `columns` option holds the
    indices of the columns the step works on (`build_pipeline` converts the
    feature names of the preset); None means all columns.
    """

    # True for steps that remove or reorder columns: a step that picks columns
    # by name must come before them, since the names no longer apply after.
    moves_columns = False

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Learn the step's state from training data. Does nothing by default.

        Args:
            x: Training features, shape (N, D).
            y: Training labels in {0, 1}, shape (N,). Most steps ignore them.

        Returns:
            The fitted step.
        """
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted step to features of shape (N, D)."""
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit on `x`, then transform it."""
        return self.fit(x, y).transform(x)


class ImputeMedian(Transform):
    """Replace NaNs with the training median of their column.

    Args:
        columns: Columns to impute; all if None. The others keep their NaNs.
        indicators: Also append, for each imputed column with NaNs in the
            training data, a column that is 1 where the value is missing.
    """

    def __init__(
        self, columns: list[int] | None = None, indicators: bool = False
    ) -> None:
        self.columns = columns
        self.indicators = indicators

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Store the median of each column, and which columns have NaNs."""
        self.columns_ = _indices(self.columns, x)
        medians, has_nan = [], []
        for j in self.columns_:
            observed = x[:, j][~np.isnan(x[:, j])]
            medians.append(np.median(observed) if len(observed) else 0.0)
            has_nan.append(len(observed) < len(x))
        self.median_ = np.array(medians)
        self.nan_columns_ = self.columns_[np.array(has_nan, dtype=bool)]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Fill NaNs with the stored medians, then append the indicators."""
        out = x.copy()
        part = out[:, self.columns_]
        out[:, self.columns_] = np.where(np.isnan(part), self.median_, part)
        if self.indicators:
            out = np.hstack([out, np.isnan(x[:, self.nan_columns_])])
        return out


class Standardize(Transform):
    """Scale each column to zero mean and unit variance.

    Constant columns are only centered, so they become 0.
    """

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Store the mean and standard deviation of each column."""
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)  # below: constant up to rounding
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Center and scale with the stored statistics."""
        return (x - self.mean_) / self.std_


class Whiten(Transform):
    """Decorrelate the columns and scale them to unit variance (PCA whitening).

    Directions whose training variance is below `rtol` times the largest one
    are dropped, so a singular covariance (constant columns, one-hot groups)
    gives fewer output columns.

    Args:
        rtol: Relative variance below which a direction is dropped.
    """

    moves_columns = True

    def __init__(self, rtol: float = 1e-9) -> None:
        self.rtol = rtol

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Store the mean and the whitening matrix of the training data."""
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        variance, directions = np.linalg.eigh(centered.T @ centered / len(x))
        keep = variance > self.rtol * variance.max()
        self.matrix_ = directions[:, keep] / np.sqrt(variance[keep])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Center and multiply by the stored whitening matrix."""
        return (x - self.mean_) @ self.matrix_


class Polynomial(Transform):
    """Append the powers 2 to `degree` of each column, without cross terms.

    Args:
        degree: Highest power added.
        columns: Columns to expand; all if None.
    """

    def __init__(self, degree: int, columns: list[int] | None = None) -> None:
        self.degree = degree
        self.columns = columns

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return `x` with the power columns appended."""
        base = x[:, _indices(self.columns, x)]
        return np.hstack([x] + [base**k for k in range(2, self.degree + 1)])


class Log1p(Transform):
    """Replace the values v of the chosen columns by log(1 + v).

    Shortens the long right tail of counts and amounts (v >= 0). NaNs stay NaN.

    Args:
        columns: Columns to transform; all if None.
    """

    def __init__(self, columns: list[int] | None = None) -> None:
        self.columns = columns

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return a copy of `x` with the chosen columns transformed."""
        out = x.copy()
        columns = _indices(self.columns, x)
        out[:, columns] = np.log1p(out[:, columns])
        return out


class OneHot(Transform):
    """Replace categorical columns by one 0/1 column per frequent value.

    For each encoded column, the values seen in at least `min_count` training
    rows get a column (the `max_levels` most frequent at most), and so does
    NaN. A row with a rarer or unseen value has 0 in all the columns of that
    feature. The columns that are not encoded come first, unchanged, followed
    by the 0/1 columns.

    Args:
        columns: Columns to encode; all if None.
        min_count: Fewest training rows for a value (or NaN) to get a column.
        max_levels: Most values of a feature that get a column, NaN aside.
    """

    moves_columns = True

    def __init__(
        self,
        columns: list[int] | None = None,
        min_count: int = 30,
        max_levels: int = 100,
    ) -> None:
        self.columns = columns
        self.min_count = min_count
        self.max_levels = max_levels

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Store, for each encoded column, the values that get a column."""
        self.columns_ = _indices(self.columns, x)
        self.levels_, self.nan_level_ = [], []
        for j in self.columns_:
            nan = np.isnan(x[:, j])
            values, counts = np.unique(x[~nan, j], return_counts=True)
            frequent = np.argsort(-counts, kind="stable")[: self.max_levels]
            frequent = frequent[counts[frequent] >= self.min_count]
            self.levels_.append(np.sort(values[frequent]))
            self.nan_level_.append(nan.sum() >= self.min_count)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return the columns not encoded, followed by the 0/1 columns."""
        kept = np.setdiff1d(np.arange(x.shape[1]), self.columns_)
        blocks = [x[:, kept]]  # sets the output dtype: the 0/1 blocks are bool
        for j, levels, nan_level in zip(self.columns_, self.levels_, self.nan_level_):
            blocks.append(x[:, j, None] == levels)
            if nan_level:
                blocks.append(np.isnan(x[:, j, None]))
        return np.hstack(blocks)


def _indices(columns: list[int] | None, x: np.ndarray) -> np.ndarray:
    """Return `columns` as an index array, or every column of `x` if None."""
    return np.arange(x.shape[1]) if columns is None else np.asarray(columns, dtype=int)


TRANSFORMS: dict[str, type[Transform]] = {
    "impute_median": ImputeMedian,
    "standardize": Standardize,
    "whiten": Whiten,
    "poly": Polynomial,
    "log1p": Log1p,
    "onehot": OneHot,
}
