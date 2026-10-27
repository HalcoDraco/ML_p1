from typing import Self

import numpy as np


class LinearModel:
    """Linear predictor on the features plus a bias column.

    Subclasses learn the weights with the matching function of
    `implementations.py`, starting from zero weights.

    Args:
        threshold: Score from which `predict` returns 1.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Add the bias column to `x` and learn the weights.

        Args:
            x: Training features, shape (N, D).
            y: Training labels in {0, 1}, shape (N,).

        Returns:
            The fitted model.
        """
        pass

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Return the linear score of each row of `x`."""
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return labels in {0, 1} by thresholding `predict_score`."""
        pass

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> np.ndarray:
        """Return the weights learned from labels `y` and features with bias `tx`."""
        pass


class LeastSquares(LinearModel):
    """Least squares with the normal equations (`least_squares`)."""

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> np.ndarray:
        """Solve the normal equations."""
        pass


class RidgeRegression(LinearModel):
    """Ridge regression (`ridge_regression`).

    Args:
        lambda_: Regularization strength.
        threshold: Score from which `predict` returns 1.
    """

    def __init__(self, lambda_: float, threshold: float = 0.5) -> None:
        pass

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> np.ndarray:
        """Solve the regularized normal equations."""
        pass


class MeanSquaredErrorGD(LinearModel):
    """Linear regression by gradient descent on the MSE (`mean_squared_error_gd`).

    Args:
        max_iters: Number of gradient steps.
        gamma: Step size.
        threshold: Score from which `predict` returns 1.
    """

    def __init__(self, max_iters: int, gamma: float, threshold: float = 0.5) -> None:
        pass

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> np.ndarray:
        """Run gradient descent from zero weights."""
        pass


class MeanSquaredErrorSGD(LinearModel):
    """Linear regression by SGD, one sample per step (`mean_squared_error_sgd`).

    Args:
        max_iters: Number of stochastic gradient steps.
        gamma: Step size.
        threshold: Score from which `predict` returns 1.
    """

    def __init__(self, max_iters: int, gamma: float, threshold: float = 0.5) -> None:
        pass

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> np.ndarray:
        """Run stochastic gradient descent from zero weights."""
        pass


class LogisticRegression(LinearModel):
    """Logistic regression by gradient descent (`logistic_regression`).

    Args:
        max_iters: Number of gradient steps.
        gamma: Step size.
        threshold: Probability from which `predict` returns 1.
    """

    def __init__(self, max_iters: int, gamma: float, threshold: float = 0.5) -> None:
        pass

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Return the predicted probability of class 1 for each row of `x`."""
        pass

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> np.ndarray:
        """Run gradient descent on the logistic loss from zero weights."""
        pass


class RegLogisticRegression(LogisticRegression):
    """L2-regularized logistic regression (`reg_logistic_regression`).

    Args:
        lambda_: Regularization strength.
        max_iters: Number of gradient steps.
        gamma: Step size.
        threshold: Probability from which `predict` returns 1.
    """

    def __init__(
        self, lambda_: float, max_iters: int, gamma: float, threshold: float = 0.5
    ) -> None:
        pass

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> np.ndarray:
        """Run gradient descent on the regularized logistic loss from zero weights."""
        pass


MODELS: dict[str, type[LinearModel]] = {
    "least_squares": LeastSquares,
    "ridge": RidgeRegression,
    "mse_gd": MeanSquaredErrorGD,
    "mse_sgd": MeanSquaredErrorSGD,
    "logistic": LogisticRegression,
    "reg_logistic": RegLogisticRegression,
}
