"""Scores of binary predictions, for labels in {0, 1} with 1 as the positive class."""

import numpy as np


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the F1 score of predicted labels, with 1 as the positive class.

    F1 = 2 TP / (2 TP + FP + FN), the harmonic mean of precision and recall.
    It is 0 when there is no positive, true or predicted.
    """
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    errors = np.sum(y_pred != y_true)  # false positives + false negatives
    denominator = 2 * true_pos + errors
    return float(2 * true_pos / denominator) if denominator else 0.0


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of correct predictions."""
    return float(np.mean(y_true == y_pred))


def roc_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Return the area under the ROC curve of `score` for labels `y` in {0, 1}.

    It is the probability that a random positive scores higher than a random
    negative (ties count one half), computed from the ranks of the scores. It
    does not depend on a threshold.
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


def best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    """Return the threshold on `score` that maximizes F1 on labels `y`.

    Predicts 1 when `score >= threshold`, trying every distinct score.
    """
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    # F1 when the rows up to each position are predicted 1:
    # 2 TP / (predicted positives + actual positives).
    true_pos = np.cumsum(y[order])
    f1 = 2 * true_pos / (np.arange(1, len(y) + 1) + y.sum())
    # A threshold predicts 1 for all the rows that share a score, so only the
    # last row of a run of ties is a possible cut.
    last_of_tie = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    return float(sorted_score[np.argmax(np.where(last_of_tie, f1, -1))])


def log_loss(score: np.ndarray, y: np.ndarray) -> float:
    """Return the mean log-loss of predicted probabilities `score` for labels `y`."""
    p = np.clip(score, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
