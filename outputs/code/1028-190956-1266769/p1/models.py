import warnings
from typing import Self

import numpy as np

import implementations
from p1.gbdt import GradientBoostedTrees
from p1.metrics import best_threshold


class LinearModel:
    """Linear predictor on the features plus a bias column.

    Subclasses learn the weights with the matching function of
    `implementations.py`, starting from zero weights.

    Args:
        threshold: Score from which `predict` returns 1, or "auto" for the
            threshold that maximizes F1 on the training data, chosen after
            fitting the weights.
    """

    def __init__(self, threshold: float | str = "auto") -> None:
        self.threshold = threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Add the bias column to `x`, learn the weights, then the threshold.

        Args:
            x: Training features, shape (N, D).
            y: Training labels in {0, 1}, shape (N,).

        Returns:
            The fitted model, with weights `w_` (bias first), the training loss
            `loss_` returned by `implementations.py`, and `threshold_`.
        """
        tx = np.c_[np.ones(len(x)), x]
        self.w_, self.loss_ = self._fit_weights(y, tx)
        if not np.isfinite(self.w_).all():
            warnings.warn(
                f"{type(self).__name__}: the weights diverged, lower gamma",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.threshold == "auto":
            self.threshold_ = best_threshold(self.predict_score(x), y)
        else:
            self.threshold_ = float(self.threshold)
        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Return the linear score of each row of `x`."""
        return x @ self.w_[1:] + self.w_[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return labels in {0, 1} by thresholding `predict_score`."""
        return (self.predict_score(x) >= self.threshold_).astype(int)

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
        """Learn the weights from labels `y` and features with bias `tx`.

        Returns:
            The weights and their training loss.
        """
        raise NotImplementedError


class LeastSquares(LinearModel):
    """Least squares with the normal equations (`least_squares`)."""

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve the normal equations."""
        return implementations.least_squares(y, tx)


class RidgeRegression(LinearModel):
    """Ridge regression (`ridge_regression`).

    Args:
        lambda_: Regularization strength.
        threshold: Score from which `predict` returns 1, or "auto".
    """

    def __init__(self, lambda_: float, threshold: float | str = "auto") -> None:
        super().__init__(threshold)
        self.lambda_ = lambda_

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve the regularized normal equations."""
        return implementations.ridge_regression(y, tx, self.lambda_)


class MeanSquaredErrorGD(LinearModel):
    """Linear regression by gradient descent on the MSE (`mean_squared_error_gd`).

    Args:
        max_iters: Number of gradient steps.
        gamma: Step size.
        threshold: Score from which `predict` returns 1, or "auto".
    """

    def __init__(
        self, max_iters: int, gamma: float, threshold: float | str = "auto"
    ) -> None:
        super().__init__(threshold)
        self.max_iters = max_iters
        self.gamma = gamma

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
        """Run gradient descent from zero weights."""
        initial_w = np.zeros(tx.shape[1])
        return implementations.mean_squared_error_gd(
            y, tx, initial_w, self.max_iters, self.gamma
        )


class MeanSquaredErrorSGD(LinearModel):
    """Linear regression by SGD, one sample per step (`mean_squared_error_sgd`).

    The samples are drawn from NumPy's global random generator, which
    `Pipeline.fit` seeds.

    Args:
        max_iters: Number of stochastic gradient steps.
        gamma: Step size.
        threshold: Score from which `predict` returns 1, or "auto".
    """

    def __init__(
        self, max_iters: int, gamma: float, threshold: float | str = "auto"
    ) -> None:
        super().__init__(threshold)
        self.max_iters = max_iters
        self.gamma = gamma

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
        """Run stochastic gradient descent from zero weights."""
        initial_w = np.zeros(tx.shape[1])
        return implementations.mean_squared_error_sgd(
            y, tx, initial_w, self.max_iters, self.gamma
        )


class LogisticRegression(LinearModel):
    """Logistic regression by gradient descent (`logistic_regression`).

    Args:
        max_iters: Number of gradient steps.
        gamma: Step size.
        threshold: Probability from which `predict` returns 1, or "auto".
    """

    def __init__(
        self, max_iters: int, gamma: float, threshold: float | str = "auto"
    ) -> None:
        super().__init__(threshold)
        self.max_iters = max_iters
        self.gamma = gamma

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Return the predicted probability of class 1 for each row of `x`."""
        return implementations.sigmoid(super().predict_score(x))

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
        """Run gradient descent on the logistic loss from zero weights."""
        initial_w = np.zeros(tx.shape[1])
        return implementations.logistic_regression(
            y, tx, initial_w, self.max_iters, self.gamma
        )


class RegLogisticRegression(LogisticRegression):
    """L2-regularized logistic regression (`reg_logistic_regression`).

    Args:
        lambda_: Regularization strength.
        max_iters: Number of gradient steps.
        gamma: Step size.
        threshold: Probability from which `predict` returns 1, or "auto".
    """

    def __init__(
        self,
        lambda_: float,
        max_iters: int,
        gamma: float,
        threshold: float | str = "auto",
    ) -> None:
        super().__init__(max_iters, gamma, threshold)
        self.lambda_ = lambda_

    def _fit_weights(self, y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
        """Run gradient descent on the regularized logistic loss from zero weights."""
        initial_w = np.zeros(tx.shape[1])
        return implementations.reg_logistic_regression(
            y, tx, self.lambda_, initial_w, self.max_iters, self.gamma
        )


# Every model has `fit`, `predict_score`, `predict` and a fitted `threshold_`.
Model = LinearModel | GradientBoostedTrees

MODELS: dict[str, type[Model]] = {
    "least_squares": LeastSquares,
    "ridge": RidgeRegression,
    "mse_gd": MeanSquaredErrorGD,
    "mse_sgd": MeanSquaredErrorSGD,
    "logistic": LogisticRegression,
    "reg_logistic": RegLogisticRegression,
    "gbdt": GradientBoostedTrees,
}
