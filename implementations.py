"""The six methods of step 2 (Table 1 of the project description).

Every method returns `(w, loss)`: the last weight vector and the loss at that
vector. The loss of the regularized methods does not include the penalty term.
`tx` is the feature matrix with the bias column already added, and all vectors
are 1-D arrays.
"""

import numpy as np


def calculate_mse(e):
    """Return the mean squared error of the error vector `e`.

    It has the factor 1/2 of the lecture notes: MSE = 1/(2N) * sum(e_n^2).
    """
    return 1 / 2 * np.mean(e**2)


def compute_gradient_linreg(y, tx, w):
    """Computes the gradient of the MSE loss at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,P)
        w: shape=(P,). The vector of model parameters.

    Returns:
        An array of shape (P,) (same shape as w), containing the gradient of the loss at w,
        and the error vector y - tx w, of shape (N,).
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.
    """

    data_size = len(y)  # Number of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def sigmoid(t):
    """Apply the logistic function 1 / (1 + exp(-t)) element-wise.

    Written as exp(-log(1 + exp(-t))), which gives the same values without
    overflowing for large negative t.
    """
    return np.exp(-np.logaddexp(0, -t))


def calculate_logistic_loss(y, tx, w):
    """Return the logistic loss (mean negative log-likelihood) of `w`.

    For labels y in {0, 1} and scores z = tx w, the loss of a sample,
    -y log(sigmoid(z)) - (1 - y) log(1 - sigmoid(z)), simplifies to
    log(1 + exp(z)) - y z. This form never takes the log of 0, so it stays
    finite when a prediction is (numerically) exactly 0 or 1.

    Args:
        y: Labels in {0, 1}, shape (N,).
        tx: Features, shape (N, D).
        w: Weights, shape (D,).
    """
    z = tx.dot(w)
    return np.mean(np.logaddexp(0, z) - y * z)


def calculate_logistic_gradient(y, tx, w):
    """Return the gradient of the logistic loss at `w`, shape (D,)."""
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm using Mean Squared Error (MSE).

    Args:
        y: shape=(N, )
        tx: shape=(N,P)
        initial_w: shape=(P, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: final model parameters as a numpy array of shape (P, )
        loss: the MSE loss at w (a scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        grad, _ = compute_gradient_linreg(y, tx, w)
        w = w - gamma * grad
    loss = calculate_mse(y - tx.dot(w))
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) using Mean Squared Error (MSE).

    Each step uses the gradient of a single data point (mini-batch size 1),
    drawn uniformly at random with NumPy's global random generator.

    Args:
        y: shape=(N, )
        tx: shape=(N,P)
        initial_w: shape=(P, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations (steps) of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: final model parameters as a numpy array of shape (P, )
        loss: the MSE loss at w on all the data (a scalar)
    """
    w = initial_w
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=max_iters):
        grad, _ = compute_gradient_linreg(y_batch, tx_batch, w)
        w = w - gamma * grad
    # The full loss costs as much as N steps, so it is computed only once, at the end.
    loss = calculate_mse(y - tx.dot(w))
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution with the normal equations.

    Solves tx^T tx w = tx^T y. The pseudo-inverse is used instead of
    `np.linalg.solve` so that a singular tx^T tx (duplicate or constant
    columns, one-hot groups) still gives a solution: the one of smallest norm.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the MSE loss at w (a scalar).
    """
    w = np.linalg.pinv(tx.T.dot(tx), hermitian=True).dot(tx.T.dot(y))
    loss = calculate_mse(y - tx.dot(w))
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression with the normal equations.

    Minimizes MSE(w) + lambda_ * ||w||^2, whose minimum solves
    (tx^T tx + 2 N lambda_ I) w = tx^T y.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, the regularization strength.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the MSE loss at w, without the penalty term (a scalar).
    """
    n, d = tx.shape
    a = tx.T.dot(tx) + 2 * n * lambda_ * np.identity(d)
    w = np.linalg.solve(a, tx.T.dot(y))
    loss = calculate_mse(y - tx.dot(w))
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent for labels in {0, 1}.

    Args:
        y: shape=(N, )
        tx: shape=(N,P)
        initial_w: shape=(P, ). The initial guess for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: final model parameters as a numpy array of shape (P, )
        loss: logistic loss at the final w
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * calculate_logistic_gradient(y, tx, w)
    loss = calculate_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent for labels in {0, 1}.

    Minimizes the logistic loss + lambda_ * ||w||^2, whose gradient is the
    logistic gradient + 2 lambda_ w.

    Args:
        y: shape=(N, )
        tx: shape=(N,P)
        lambda_: a scalar denoting the L2 regularization strength
        initial_w: shape=(P, ). The initial guess for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: final model parameters as a numpy array of shape (P, )
        loss: logistic loss at the final w, without the penalty term
    """
    w = initial_w
    for _ in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = calculate_logistic_loss(y, tx, w)
    return w, loss
