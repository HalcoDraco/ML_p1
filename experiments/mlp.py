"""Scratch: NumPy MLP for tabular data, plus its input encoding.

- `MLPEncoding`: numeric features -> piecewise-linear encoding (PLE) on
  quantile bins plus a NaN indicator (Gorishniy et al. 2022); categorical
  features -> one-hot with a NaN column; then every column standardized.
- `MLP`: ReLU MLP trained with AdamW on the log-loss, dropout, cosine
  learning-rate decay, early stopping on an inner holdout; `n_models` nets
  with different seeds and holdouts, averaged on the logit scale.
"""

import numpy as np

from p1 import transforms as T
from p1.metrics import best_threshold


class MLPEncoding(T.Transform):
    moves_columns = True

    def __init__(self, numeric=None, n_bins=16, min_count=30, max_levels=100):
        self.numeric = numeric
        self.n_bins = n_bins
        self.min_count = min_count
        self.max_levels = max_levels

    def fit(self, x, y=None):
        d = x.shape[1]
        self.numeric_ = np.asarray(
            self.numeric if self.numeric is not None else [], int
        )
        self.categorical_ = np.setdiff1d(np.arange(d), self.numeric_)
        self.edges_ = []
        for j in self.numeric_:
            col = x[:, j][~np.isnan(x[:, j])]
            edges = np.unique(np.quantile(col, np.linspace(0, 1, self.n_bins + 1)))
            self.edges_.append(edges)
        self.onehot_ = T.OneHot(
            columns=list(range(len(self.categorical_))),
            min_count=self.min_count,
            max_levels=self.max_levels,
        ).fit(x[:, self.categorical_])
        raw = self._raw(x)
        self.mean_ = raw.mean(axis=0)
        std = raw.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0).astype(np.float32)
        return self

    def _raw(self, x):
        blocks = []
        for j, edges in zip(self.numeric_, self.edges_):
            col = x[:, j]
            nan = np.isnan(col)
            if len(edges) < 2:
                blocks.append(nan[:, None].astype(np.float32))
                continue
            lo, hi = edges[:-1], edges[1:]
            ple = np.clip((col[:, None] - lo) / (hi - lo), 0, 1)
            ple[nan] = 0
            blocks.append(ple.astype(np.float32))
            blocks.append(nan[:, None].astype(np.float32))
        blocks.append(
            self.onehot_.transform(x[:, self.categorical_]).astype(np.float32)
        )
        return np.hstack(blocks)

    def transform(self, x):
        out = self._raw(x)
        out -= self.mean_
        out /= self.std_
        return out


def _sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


class MLP:
    def __init__(
        self,
        hidden=(512, 256),
        dropout=0.2,
        input_dropout=0.0,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=512,
        epochs=8,
        valid_fraction=0.1,
        n_models=1,
        seed=0,
        threshold="auto",
        verbose=True,
    ):
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_fraction = valid_fraction
        self.n_models = n_models
        self.seed = seed
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, x, y):
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.nets_, self.best_epochs_ = [], []
        for m in range(self.n_models):
            net, epoch = self._fit_one(
                x, y, np.random.default_rng(self.seed * 1000 + m)
            )
            self.nets_.append(net)
            self.best_epochs_.append(epoch)
        if self.threshold == "auto":
            self.threshold_ = best_threshold(self.predict_score(x), y)
        else:
            self.threshold_ = float(self.threshold)
        return self

    def _init(self, d, rng, prior):
        sizes = [d, *self.hidden, 1]
        params = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6 / fan_in) if fan_out > 1 else np.sqrt(1 / fan_in)
            w = rng.uniform(-bound, bound, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            params += [w, b]
        params[-1][:] = np.log(prior / (1 - prior))
        return params

    def _forward(self, params, xb, rng=None):
        """Return the logits and, when training (rng given), the cache for backprop."""
        cache = []
        h = xb
        if rng is not None and self.input_dropout > 0:
            mask = rng.random(h.shape, dtype=np.float32) >= self.input_dropout
            h = h * mask / (1 - self.input_dropout)
        n_layers = len(params) // 2
        for i in range(n_layers):
            w, b = params[2 * i], params[2 * i + 1]
            z = h @ w + b
            if i == n_layers - 1:
                cache.append((h, None, None))
                return z[:, 0], cache
            a = np.maximum(z, 0)
            mask = None
            if rng is not None and self.dropout > 0:
                mask = (
                    rng.random(a.shape, dtype=np.float32) >= self.dropout
                ) / np.float32(1 - self.dropout)
                a = a * mask
            cache.append((h, z, mask))
            h = a

    def _backward(self, params, cache, g):
        """g: dL/dlogit, shape (B,). Returns the gradients, same order as params."""
        grads = [None] * len(params)
        n_layers = len(params) // 2
        gz = g[:, None]
        for i in reversed(range(n_layers)):
            h, _, _ = cache[i]
            w = params[2 * i]
            grads[2 * i] = h.T @ gz
            grads[2 * i + 1] = gz.sum(axis=0)
            if i == 0:
                break
            gh = gz @ w.T
            _, z_prev, mask_prev = cache[i - 1]
            if mask_prev is not None:
                gh *= mask_prev
            gz = gh * (z_prev > 0)
        return grads

    def _logits(self, params, x):
        out = np.empty(len(x), dtype=np.float32)
        for s in range(0, len(x), 8192):
            out[s : s + 8192] = self._forward(params, x[s : s + 8192])[0]
        return out

    def _fit_one(self, x, y, rng):
        n, d = x.shape
        perm = rng.permutation(n)
        n_valid = int(self.valid_fraction * n)
        valid, train = perm[:n_valid], perm[n_valid:]
        params = self._init(d, rng, float(np.clip(y[train].mean(), 1e-4, 1 - 1e-4)))
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        steps_per_epoch = int(np.ceil(len(train) / self.batch_size))
        total = steps_per_epoch * self.epochs
        step = 0
        best, best_loss, best_epoch = None, np.inf, 0
        for epoch in range(self.epochs):
            order = rng.permutation(train)
            for s in range(0, len(order), self.batch_size):
                rows = np.sort(order[s : s + self.batch_size])
                xb, yb = x[rows], y[rows]
                z, cache = self._forward(params, xb, rng)
                g = (_sigmoid(z) - yb) / len(rows)
                grads = self._backward(params, cache, g.astype(np.float32))
                lr = self.lr * 0.5 * (1 + np.cos(np.pi * step / total))
                step += 1
                c1, c2 = 1 - b1**step, 1 - b2**step
                for p, gr, mi, vi in zip(params, grads, m, v):
                    mi *= b1
                    mi += (1 - b1) * gr
                    vi *= b2
                    vi += (1 - b2) * gr * gr
                    if p.ndim == 2 and self.weight_decay > 0:
                        p *= 1 - lr * self.weight_decay
                    p -= (lr / c1) * mi / (np.sqrt(vi / c2) + eps)
            if n_valid:
                zv = self._logits(params, x[valid])
                pv = np.clip(_sigmoid(zv), 1e-7, 1 - 1e-7)
                loss = -np.mean(y[valid] * np.log(pv) + (1 - y[valid]) * np.log(1 - pv))
                if self.verbose:
                    print(
                        f"    epoch {epoch + 1}: inner-valid log-loss {loss:.5f}",
                        flush=True,
                    )
                if loss < best_loss:
                    best_loss, best_epoch = loss, epoch + 1
                    best = [p.copy() for p in params]
            else:
                best, best_epoch = params, epoch + 1
        return best, best_epoch

    def predict_logit(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        return np.mean([self._logits(p, x) for p in self.nets_], axis=0)

    def predict_score(self, x):
        return _sigmoid(self.predict_logit(x).astype(np.float64))

    def predict(self, x):
        return (self.predict_score(x) >= self.threshold_).astype(int)
