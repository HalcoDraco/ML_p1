from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from p1.paths import FIGURES_DIR

# Colorblind-safe series colors, and ink colors for text, axes and grid.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def use_style() -> None:
    """Set a light, uncluttered matplotlib style for the notebook figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c3c2b7",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SOFT,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def missing_fraction(x: np.ndarray) -> np.ndarray:
    """Return the fraction of NaN values in each column of `x`."""
    return np.isnan(x).mean(axis=0)


def positive_rate_by_value(
    values: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count the rows and the fraction of positives for each value of a feature.

    Args:
        values: One feature column, shape (N,).
        y: Labels in {0, 1}, shape (N,).

    Returns:
        The distinct values (NaN last, if present), their row counts and their
        positive rates.
    """
    observed = ~np.isnan(values)
    levels, inverse, counts = np.unique(
        values[observed], return_inverse=True, return_counts=True
    )
    rates = np.bincount(inverse, weights=y[observed]) / counts
    if not observed.all():
        levels = np.append(levels, np.nan)
        counts = np.append(counts, (~observed).sum())
        rates = np.append(rates, y[~observed].mean())
    return levels, counts, rates


def roc_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Return the area under the ROC curve of `score` for labels `y` in {0, 1}.

    It is the probability that a random positive scores higher than a random
    negative (ties count one half), computed from the ranks of the scores.
    """
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    is_new = np.r_[True, sorted_score[1:] != sorted_score[:-1]]
    group = np.cumsum(is_new) - 1
    group_size = np.bincount(group)
    mean_rank = np.cumsum(group_size) - (group_size - 1) / 2  # ties share a rank
    ranks = np.empty(len(score))
    ranks[order] = mean_rank[group]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def univariate_auc(
    x: np.ndarray, y: np.ndarray, max_levels: int = 50, n_bins: int = 20, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Measure how well each feature alone separates the two classes.

    Two AUCs per column, both folded so that 0.5 means no signal and 1 means
    perfect separation:
    - encoded: the column is split into categories (its values if it has at
      most `max_levels` of them, quantile bins otherwise, NaN as one more
      category), and each row is scored by the positive rate of its category
      in the other half of the data. This catches any shape of relation,
      without rewarding categories that are too small to generalize.
    - raw: the column's values used directly as scores, NaN set to the median.
      This is what a linear model sees without any encoding.

    Args:
        x: Features, shape (N, D).
        y: Labels in {0, 1}, shape (N,).
        max_levels: Most distinct values a column can have to keep them as categories.
        n_bins: Number of quantile bins for the other columns.
        seed: Seed of the split into halves.

    Returns:
        The encoded and the raw AUC of each column, each of shape (D,).
    """
    halves = np.array_split(np.random.default_rng(seed).permutation(len(y)), 2)
    prior = y.mean()
    encoded, raw = np.empty(x.shape[1]), np.empty(x.shape[1])
    for j in range(x.shape[1]):
        col = x[:, j]
        observed = ~np.isnan(col)
        category = np.zeros(len(col), dtype=int)  # 0 is NaN
        levels = np.unique(col[observed])
        if len(levels) > max_levels:
            quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
            levels = np.unique(np.quantile(col[observed], quantiles))
            category[observed] = 1 + np.searchsorted(levels, col[observed], "right")
        else:
            category[observed] = 1 + np.searchsorted(levels, col[observed])
        score = np.empty(len(y))
        for fit, apply in (halves, halves[::-1]):
            sums = np.bincount(category[fit], weights=y[fit], minlength=len(levels) + 2)
            counts = np.bincount(category[fit], minlength=len(levels) + 2)
            # Shrink small categories toward the overall rate (20 pseudo-rows).
            score[apply] = ((sums + 20 * prior) / (counts + 20))[category[apply]]
        encoded[j] = roc_auc(score, y)
        median = np.median(col[observed]) if observed.any() else 0.0
        raw[j] = roc_auc(np.where(observed, col, median), y)
    return np.maximum(encoded, 1 - encoded), np.maximum(raw, 1 - raw)


def duplicate_columns(x: np.ndarray, names: list[str]) -> list[tuple[str, str]]:
    """Return the pairs of columns with identical values, NaN included."""
    first_seen, pairs = {}, []
    for j, name in enumerate(names):
        key = np.nan_to_num(x[:, j], nan=np.inf).tobytes()
        if key in first_seen:
            pairs.append((first_seen[key], name))
        else:
            first_seen[key] = name
    return pairs


def correlated_pairs(
    x: np.ndarray, names: list[str], threshold: float = 0.95, min_rows: int = 1000
) -> list[tuple[float, str, str]]:
    """Find pairs of columns whose correlation exceeds `threshold` in absolute value.

    Each correlation uses the rows where both columns are observed, and pairs
    observed together in fewer than `min_rows` rows are skipped.

    Returns:
        `(|r|, name_a, name_b)` triples, sorted from the highest |r|.
    """
    observed = (~np.isnan(x)).astype(float)
    sd = np.nanstd(x, axis=0)
    z = np.nan_to_num((x - np.nanmean(x, axis=0)) / np.where(sd > 0, sd, 1))
    n = observed.T @ observed  # rows where both columns are observed
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_a = (z.T @ observed) / n  # mean of column a over those rows
        var_a = ((z**2).T @ observed) / n - mean_a**2
        cov = (z.T @ z) / n - mean_a * mean_a.T
        r = cov / np.sqrt(var_a * var_a.T)
    a, b = np.triu_indices(len(names), 1)
    keep = (n[a, b] >= min_rows) & (np.abs(r[a, b]) > threshold) & np.isfinite(r[a, b])
    pairs = [(abs(r[i, j]), names[i], names[j]) for i, j in zip(a[keep], b[keep])]
    return sorted(pairs, reverse=True)


def onehot_encode(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    max_levels: int = 30,
    n_bins: int = 10,
    min_count: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode every column, with the categories learned on `x_train`.

    A column with at most `max_levels` distinct values gets one 0/1 column per
    value seen at least `min_count` times; other columns are first cut into
    `n_bins` quantile bins. A column with NaNs also gets a NaN indicator. Codes
    such as "don't know" or "none" therefore get their own columns without any
    knowledge of the codebook.

    Returns:
        The encoded `x_train` and `x_valid`, as float32 arrays.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    plans = []  # per column: bin edges (or None), kept levels, NaN indicator
    for col in x_train.T:
        observed = col[~np.isnan(col)]
        edges = None
        levels, counts = np.unique(observed, return_counts=True)
        if len(levels) > max_levels:
            edges = np.unique(np.quantile(observed, quantiles))
            binned = np.searchsorted(edges, observed, "right")
            levels, counts = np.unique(binned, return_counts=True)
        plans.append((edges, levels[counts >= min_count], len(observed) < len(col)))
    width = sum(len(levels) + has_nan for _, levels, has_nan in plans)

    def encode(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), width), dtype=np.float32)
        k = 0
        for col, (edges, levels, has_nan) in zip(x.T, plans):
            if edges is not None:
                col = np.where(
                    np.isnan(col), np.nan, np.searchsorted(edges, col, "right")
                )
            out[:, k : k + len(levels)] = col[:, None] == levels
            k += len(levels)
            if has_nan:
                out[:, k] = np.isnan(col)
                k += 1
        return out

    return encode(x_train), encode(x_valid)


def ridge_probe(
    x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, lambda_: float = 1e-3
) -> np.ndarray:
    """Score `x_valid` with a ridge regression fitted on standardized `x_train`.

    A quick check of how much signal a linear model finds in a feature matrix.
    The normal equations are written from the Gram matrix of `x_train`, so the
    standardized copy of a large matrix is never built.

    Args:
        x_train: Training features without NaNs, shape (N, D).
        y_train: Training labels in {0, 1}, shape (N,).
        x_valid: Features to score, shape (M, D).
        lambda_: Ridge penalty on the standardized weights.

    Returns:
        The score of each row of `x_valid`, roughly a probability of class 1.
    """
    n = len(y_train)
    mean = x_train.sum(axis=0, dtype=float) / n
    cov = (x_train.T @ x_train).astype(float) / n - np.outer(mean, mean)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    std[std == 0] = 1
    corr = cov / np.outer(std, std)
    target = (x_train.T @ y_train / n - mean * y_train.mean()) / std
    w = np.linalg.solve(corr + lambda_ * np.eye(len(std)), target) / std
    return y_train.mean() + x_valid @ w.astype(x_valid.dtype) - mean @ w


def f1_by_threshold(
    score: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the thresholds and the precision, recall and F1 of each.

    Predicting 1 when `score >= threshold`, with every distinct score tried as
    a threshold, from the highest to the lowest.
    """
    order = np.argsort(-score, kind="stable")
    true_pos = np.cumsum(y[order])
    predicted_pos = np.arange(1, len(y) + 1)
    last_of_tie = np.r_[score[order][1:] != score[order][:-1], True]
    true_pos, predicted_pos = true_pos[last_of_tie], predicted_pos[last_of_tie]
    precision = true_pos / predicted_pos
    recall = true_pos / y.sum()
    f1 = 2 * true_pos / (predicted_pos + y.sum())
    return score[order][last_of_tie], precision, recall, f1


def print_table(header: list[str], rows: list[tuple], align: str = "") -> None:
    """Print rows as an aligned text table; floats get 3 decimals.

    Args:
        header: Column titles.
        rows: One tuple of values per row.
        align: One letter per column, "l" (left) or "r" (right). By default
            the first column is left-aligned and the others right-aligned.
    """
    align = align or "l" + "r" * (len(header) - 1)
    cells = [
        [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row] for row in rows
    ]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(header)]

    def line(values: list[str]) -> str:
        return "  ".join(
            v.ljust(w) if a == "l" else v.rjust(w)
            for v, w, a in zip(values, widths, align)
        ).rstrip()

    print(line(header))
    for row in cells:
        print(line(row))


def save_figure(fig: Figure, name: str) -> Path:
    """Save `fig` as `report/figures/<name>.pdf` and return the file path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    return path
