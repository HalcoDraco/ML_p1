"""Gradient-boosted decision trees on binned features, in NumPy.

The trees are grown level by level from histograms of the gradients, as in
LightGBM (Ke et al. 2017), with the Newton leaf values and split gains of
XGBoost (Chen & Guestrin 2016). Each feature is binned once at the start of
`fit`. Bin 0 holds the missing values, so every split also learns on which
side they go.
"""

from typing import Self

import numpy as np

from p1.metrics import best_threshold


class GradientBoostedTrees:
    """Binary classifier: a sum of regression trees fitted to the log-loss.

    Every tree is complete, of depth `depth`, and each of its splits sends a
    set of bins of one feature to the left child:
    - a numeric feature splits between two consecutive bins, and its missing
      values go to the side where they gain more;
    - a categorical feature sorts its bins (missing values included) by
      gradient over hessian and splits that order once, which finds the best
      grouping of its levels in two (Fisher 1958).
    A node without a split of positive gain sends all its rows left.

    Args:
        n_trees: Number of boosting rounds.
        learning_rate: Shrinkage of every leaf value.
        depth: Depth of every tree, which has 2**depth leaves.
        lambda_: L2 penalty on the leaf values.
        min_child_hessian: Smallest hessian sum allowed in a child node.
        row_fraction: Fraction of the rows sampled for each tree.
        feature_fraction: Fraction of the features sampled for each tree.
        max_bins: Bins per feature, the missing-value bin included; at most 256.
        min_count: Training rows a categorical level needs to get its own bin;
            rarer and unseen levels share one.
        cat_smooth: Added to the hessian sums when sorting categorical bins,
            which pulls rare bins towards the middle of the order.
        categorical: Indices of the categorical features; the others are numeric.
        seed: Seed of the row and feature sampling.
        threshold: Probability from which `predict` returns 1, or "auto" for
            the threshold that maximizes F1 on the training rows.
    """

    def __init__(
        self,
        n_trees: int = 500,
        learning_rate: float = 0.05,
        depth: int = 5,
        lambda_: float = 1.0,
        min_child_hessian: float = 5.0,
        row_fraction: float = 0.7,
        feature_fraction: float = 0.6,
        max_bins: int = 64,
        min_count: int = 30,
        cat_smooth: float = 10.0,
        categorical: list[int] | None = None,
        seed: int = 0,
        threshold: float | str = "auto",
    ) -> None:
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.depth = depth
        self.lambda_ = lambda_
        self.min_child_hessian = min_child_hessian
        self.row_fraction = row_fraction
        self.feature_fraction = feature_fraction
        self.max_bins = max_bins
        self.min_count = min_count
        self.cat_smooth = cat_smooth
        self.categorical = categorical
        self.seed = seed
        self.threshold = threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Bin the features, then add the trees one by one.

        Args:
            x: Training features, NaN for missing values, shape (N, D).
            y: Training labels in {0, 1}, shape (N,).

        Returns:
            The fitted model, with the bins (`cuts_`), the trees (`features_`,
            `masks_`, `values_`), the initial log-odds `base_score_`, the summed
            split gain of each feature `importance_`, and `threshold_`.
        """
        y = y.astype(float)
        self._fit_bins(x)
        xb = self._bin(x)
        n, d = xb.shape
        rng = np.random.default_rng(self.seed)
        n_features = max(1, round(self.feature_fraction * d))
        self.features_ = np.zeros((self.n_trees, 2**self.depth - 1), dtype=np.intp)
        self.masks_ = np.ones(
            (self.n_trees, 2**self.depth - 1, self.max_bins), dtype=bool
        )
        self.values_ = np.zeros((self.n_trees, 2**self.depth))
        self.importance_ = np.zeros(d)
        rate = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_score_ = float(np.log(rate / (1 - rate)))
        score = np.full(n, self.base_score_)
        for t in range(self.n_trees):
            p = np.clip(_sigmoid(score), 1e-6, 1 - 1e-6)
            rows = np.flatnonzero(rng.random(n) < self.row_fraction)
            features = np.sort(rng.choice(d, n_features, replace=False))
            grad = p[rows] - y[rows]
            hess = p[rows] * (1 - p[rows])
            self._grow(t, xb, rows, features, grad, hess)
            score += self.values_[t, self._leaves(t, xb)]
        if self.threshold == "auto":
            self.threshold_ = best_threshold(_sigmoid(score), y)
        else:
            self.threshold_ = float(self.threshold)
        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Return the predicted probability of class 1 for each row of `x`."""
        return self.staged_scores(x, every=self.n_trees)[-1][1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return labels in {0, 1} by thresholding `predict_score`."""
        return (self.predict_score(x) >= self.threshold_).astype(int)

    def staged_scores(
        self, x: np.ndarray, every: int = 50
    ) -> list[tuple[int, np.ndarray]]:
        """Return the probabilities after every `every` trees and after the last.

        Returns:
            Pairs of (number of trees, probabilities of the rows of `x`).
        """
        xb = self._bin(x)
        score = np.full(len(x), self.base_score_)
        stages = []
        for t in range(len(self.values_)):
            score += self.values_[t, self._leaves(t, xb)]
            if (t + 1) % every == 0 or t + 1 == len(self.values_):
                stages.append((t + 1, _sigmoid(score)))
        return stages

    def _fit_bins(self, x: np.ndarray) -> None:
        """Choose the bins of each feature from the training rows.

        A numeric feature gets one bin per value if it has fewer values than
        bins, and quantile bins otherwise; `cuts_[j]` holds the boundaries. A
        categorical feature gets one bin per level with at least `min_count`
        rows (the `max_bins` - 2 most frequent at most) and one shared by the
        other levels; `cuts_[j]` holds the levels with a bin.
        """
        d = x.shape[1]
        self.is_categorical_ = np.zeros(d, dtype=bool)
        if self.categorical is not None:
            self.is_categorical_[self.categorical] = True
        self.cuts_ = []
        for j in range(d):
            col = x[:, j]
            col = col[~np.isnan(col)]
            values, counts = np.unique(col, return_counts=True)
            if self.is_categorical_[j]:
                frequent = np.flatnonzero(counts >= self.min_count)
                by_count = np.argsort(-counts[frequent], kind="stable")
                frequent = frequent[by_count][: self.max_bins - 2]
                self.cuts_.append(np.sort(values[frequent]))
            elif len(values) < self.max_bins:
                self.cuts_.append((values[:-1] + values[1:]) / 2)
            else:
                quantiles = np.linspace(0, 1, self.max_bins)[1:-1]
                self.cuts_.append(np.unique(np.quantile(col, quantiles)))

    def _bin(self, x: np.ndarray) -> np.ndarray:
        """Return the bin of every value of `x`, as uint8 in column-major order."""
        xb = np.empty(x.shape, dtype=np.uint8, order="F")
        for j, cuts in enumerate(self.cuts_):
            col = x[:, j]
            if not self.is_categorical_[j]:
                b = np.searchsorted(cuts, col, side="right") + 1
            elif len(cuts):
                pos = np.searchsorted(cuts, col)
                found = cuts[np.minimum(pos, len(cuts) - 1)] == col
                b = np.where(found, pos + 1, len(cuts) + 1)
            else:
                b = np.ones(len(col), dtype=np.intp)
            b[np.isnan(col)] = 0
            xb[:, j] = b
        return xb

    def _grow(
        self,
        t: int,
        xb: np.ndarray,
        rows: np.ndarray,
        features: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
    ) -> None:
        """Grow tree `t` level by level on the sampled rows and features.

        Args:
            t: Index of the tree.
            xb: Binned training features.
            rows: Indices of the sampled rows.
            features: Indices of the sampled features, sorted.
            grad: Gradient of the log-loss of each sampled row.
            hess: Hessian of the log-loss of each sampled row.
        """
        bins = self.max_bins
        xs = np.empty((len(rows), len(features)), dtype=np.uint8, order="F")
        for i, j in enumerate(features):
            xs[:, i] = xb[:, j][rows]
        categorical = self.is_categorical_[features]
        node = np.zeros(len(rows), dtype=np.intp)  # node of each row in its level
        hist_g, hist_h = _histograms(xs, None, node, grad, hess, 1, bins)
        for level in range(self.depth):
            n_nodes = 2**level
            if level > 0:
                hist_g, hist_h = self._child_histograms(
                    xs, node, grad, hess, hist_g, hist_h
                )
            feature, mask, gain = self._best_splits(hist_g, hist_h, categorical)
            first = n_nodes - 1  # nodes are stored level by level
            self.features_[t, first : first + n_nodes] = features[feature]
            self.masks_[t, first : first + n_nodes] = mask
            np.add.at(self.importance_, features[feature], gain)
            go_right = ~mask[node, xs[np.arange(len(rows)), feature[node]]]
            node = 2 * node + go_right
        leaf_g = np.bincount(node, weights=grad, minlength=2**self.depth)
        leaf_h = np.bincount(node, weights=hess, minlength=2**self.depth)
        self.values_[t] = -self.learning_rate * leaf_g / (leaf_h + self.lambda_)

    def _child_histograms(
        self,
        xs: np.ndarray,
        node: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        parent_g: np.ndarray,
        parent_h: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the histograms of a level from those of the level above.

        Only the child with fewer rows of each parent is counted; its sibling
        is the parent minus that child (Ke et al. 2017). Splits are often
        lopsided, so this counts far fewer than half of the rows.

        Args:
            xs: Binned sampled rows, shape (rows, sampled features).
            node: Node of each row in the new level.
            grad: Gradient of each row.
            hess: Hessian of each row.
            parent_g: Gradient sums of the level above, (features, parents, bins).
            parent_h: Hessian sums of the level above, same shape.

        Returns:
            The gradient and hessian sums, shape (features, 2 * parents, bins).
        """
        n_features, n_parents, bins = parent_g.shape
        size = np.bincount(node, minlength=2 * n_parents)
        right_smaller = size[1::2] < size[0::2]
        small = np.flatnonzero((node % 2 == 1) == right_smaller[node // 2])
        small_g, small_h = _histograms(
            xs, small, node[small] // 2, grad[small], hess[small], n_parents, bins
        )
        swap = right_smaller[:, None]
        children = []
        for small_hist, parent in ((small_g, parent_g), (small_h, parent_h)):
            big = parent - small_hist
            left = np.where(swap, big, small_hist)
            right = np.where(swap, small_hist, big)
            # Interleave, so that the children of parent k are nodes 2k and 2k + 1.
            stacked = np.stack([left, right], axis=2)
            children.append(stacked.reshape(n_features, 2 * n_parents, bins))
        return children[0], children[1]

    def _best_splits(
        self, hist_g: np.ndarray, hist_h: np.ndarray, categorical: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the best split of every node of a level.

        Every candidate split sends a prefix of an order of the bins to the
        left: the natural order with the missing-value bin first or last for
        numeric features, the order of gradient over hessian for categorical
        ones. The gain of a split is G_L²/(H_L + λ) + G_R²/(H_R + λ) - G²/(H + λ).

        Args:
            hist_g: Gradient sums, shape (sampled features, nodes, bins).
            hist_h: Hessian sums, same shape.
            categorical: Whether each sampled feature is categorical.

        Returns:
            For each node, the index of the split feature among the sampled
            ones, the mask of the bins sent left, and the gain (0 and all bins
            left when no split has a positive gain).
        """
        _, n_nodes, bins = hist_g.shape
        lam, min_hessian = self.lambda_, self.min_child_hessian
        total_g = hist_g[0].sum(axis=1)[:, None]  # every feature sums to the node
        total_h = hist_h[0].sum(axis=1)[:, None]
        parent = total_g**2 / (total_h + lam)
        best_gain = np.zeros(n_nodes)
        best_feature = np.zeros(n_nodes, dtype=np.intp)
        best_mask = np.ones((n_nodes, bins), dtype=bool)
        numeric = np.flatnonzero(~categorical)
        nominal = np.flatnonzero(categorical)
        shape = (len(numeric), n_nodes, bins)
        candidates = [
            (numeric, np.broadcast_to(np.arange(bins), shape)),  # missing go left
            (numeric, np.broadcast_to(np.r_[1:bins, 0], shape)),  # missing go right
            (
                nominal,
                np.argsort(
                    hist_g[nominal] / (hist_h[nominal] + self.cat_smooth), axis=2
                ),
            ),
        ]
        for members, order in candidates:
            if len(members) == 0:
                continue
            left_g = np.take_along_axis(hist_g[members], order, axis=2).cumsum(axis=2)
            left_h = np.take_along_axis(hist_h[members], order, axis=2).cumsum(axis=2)
            left_g, left_h = left_g[:, :, :-1], left_h[:, :, :-1]
            right_g, right_h = total_g - left_g, total_h - left_h
            gain = left_g**2 / (left_h + lam) + right_g**2 / (right_h + lam) - parent
            gain[(left_h < min_hessian) | (right_h < min_hessian)] = -np.inf
            gain = gain.transpose(1, 0, 2).reshape(n_nodes, -1)
            arg = gain.argmax(axis=1)
            node_gain = gain[np.arange(n_nodes), arg]
            which, cut = np.divmod(arg, bins - 1)
            for k in np.flatnonzero(node_gain > best_gain):
                best_gain[k] = node_gain[k]
                best_feature[k] = members[which[k]]
                best_mask[k] = False
                best_mask[k, order[which[k], k, : cut[k] + 1]] = True
        return best_feature, best_mask, best_gain

    def _leaves(self, t: int, xb: np.ndarray) -> np.ndarray:
        """Return the leaf of tree `t` that each row of binned features reaches."""
        rows = np.arange(len(xb))
        node = np.zeros(len(xb), dtype=np.intp)
        for level in range(self.depth):
            heap = 2**level - 1 + node  # index of the node in the tree's arrays
            b = xb[rows, self.features_[t, heap]]
            node = 2 * node + ~self.masks_[t, heap, b]
        return node


def _histograms(
    xs: np.ndarray,
    rows: np.ndarray | None,
    node: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    n_nodes: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the gradient and hessian sums of every (feature, node, bin).

    Args:
        xs: Binned sampled rows, shape (rows, features), column-major.
        rows: Rows of `xs` to count, all if None.
        node: Node of each counted row, below `n_nodes`.
        grad: Gradient of each counted row.
        hess: Hessian of each counted row.
        n_nodes: Number of nodes.
        bins: Number of bins per feature.

    Returns:
        Two arrays of shape (features, nodes, bins).
    """
    n_features, size = xs.shape[1], n_nodes * bins
    offset = node * bins
    hist_g = np.empty((n_features, size))
    hist_h = np.empty((n_features, size))
    for i in range(n_features):
        col = xs[:, i] if rows is None else xs[:, i][rows]
        index = offset + col
        hist_g[i] = np.bincount(index, weights=grad, minlength=size)
        hist_h[i] = np.bincount(index, weights=hess, minlength=size)
    shape = (n_features, n_nodes, bins)
    return hist_g.reshape(shape), hist_h.reshape(shape)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Return 1 / (1 + exp(-z)), computed without overflow."""
    return 0.5 * (1 + np.tanh(0.5 * z))
