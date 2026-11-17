"""Segment cost functions with O(1) evaluation from prefix sums.

Every cost is twice the negative maximised log likelihood of a segment
(constants that do not depend on the segmentation are dropped). Using the
same scale for every cost means the BIC style penalties in ``penalties.py``
mean the same thing whichever cost you pick.

A segment is the half open slice ``y[start:end]``. All costs expose a
vectorised ``cost_pairs(starts, ends)`` which the search algorithms use to
score many candidate segments in one NumPy call.

Every cost here is the minimum over a parameter set of a sum of per point
terms, so splitting a segment can never increase total cost:

    C(a, b) >= C(a, s) + C(s, b)   for a < s < b

PELT's pruning rule relies on exactly this property.
"""

from __future__ import annotations

import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))


def _as_2d(y) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("signal must be 1 or 2 dimensional (n,) or (n, d)")
    if arr.shape[0] == 0:
        raise ValueError("signal is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal contains NaN or infinite values")
    return arr


def _prefix(arr: np.ndarray) -> np.ndarray:
    out = np.zeros((arr.shape[0] + 1, arr.shape[1]))
    np.cumsum(arr, axis=0, out=out[1:])
    return out


def mad_sigma(y) -> np.ndarray:
    """Robust per column noise scale from first differences.

    Differencing cancels piecewise constant means except at the (few)
    changepoints, and the median absolute deviation ignores those outliers.
    Dividing by sqrt(2) undoes the variance doubling caused by differencing.
    """
    arr = _as_2d(y)
    if arr.shape[0] < 3:
        return np.ones(arr.shape[1])
    d = np.diff(arr, axis=0)
    med = np.median(d, axis=0)
    sigma = 1.4826 * np.median(np.abs(d - med), axis=0) / np.sqrt(2.0)
    fallback = arr.std(axis=0)
    sigma = np.where(sigma > 0, sigma, np.where(fallback > 0, fallback, 1.0))
    return sigma


class Cost:
    """Base class. Subclasses implement ``fit`` and ``cost_pairs``."""

    name = "base"
    min_size = 1
    params_per_dim = 1

    def __init__(self) -> None:
        self.n = 0
        self.d = 0

    @property
    def n_params(self) -> int:
        """Free parameters per segment (used by BIC/AIC penalties)."""
        return self.params_per_dim * self.d

    def fit(self, y) -> "Cost":
        raise NotImplementedError

    def cost_pairs(self, starts, ends) -> np.ndarray:
        raise NotImplementedError

    def cost_many(self, starts, end: int) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int64)
        return self.cost_pairs(starts, np.full(starts.shape, end, dtype=np.int64))

    def cost(self, start: int, end: int) -> float:
        if not 0 <= start < end <= self.n:
            raise IndexError(f"bad segment [{start}, {end}) for n={self.n}")
        return float(self.cost_pairs(np.array([start]), np.array([end]))[0])

    def total(self, changepoints) -> float:
        """Unpenalised cost of the segmentation given by ``changepoints``."""
        bounds = [0, *sorted(int(c) for c in changepoints), self.n]
        starts = np.array(bounds[:-1])
        ends = np.array(bounds[1:])
        return float(self.cost_pairs(starts, ends).sum())


class L2Cost(Cost):
    """Change in mean, Gaussian noise with known (or estimated) scale.

    cost = sum over columns of SSE / sigma^2.
    """

    name = "l2"
    min_size = 1
    params_per_dim = 1

    def __init__(self, sigma=None) -> None:
        super().__init__()
        self.sigma = sigma

    def fit(self, y) -> "L2Cost":
        arr = _as_2d(y)
        self.n, self.d = arr.shape
        sigma = mad_sigma(arr) if self.sigma is None else np.broadcast_to(
            np.asarray(self.sigma, dtype=float), (self.d,)
        )
        if np.any(sigma <= 0):
            raise ValueError("sigma must be positive")
        self.sigma_ = np.asarray(sigma, dtype=float)
        self._s1 = _prefix(arr)
        self._s2 = _prefix(arr * arr)
        self._w = 1.0 / self.sigma_ ** 2
        return self

    def cost_pairs(self, starts, ends) -> np.ndarray:
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        length = (ends - starts)[:, None].astype(float)
        s1 = self._s1[ends] - self._s1[starts]
        s2 = self._s2[ends] - self._s2[starts]
        sse = np.maximum(s2 - s1 * s1 / length, 0.0)
        return sse @ self._w


class NormalMeanVarCost(Cost):
    """Change in mean and variance (independent Gaussian columns).

    The variance estimate is floored at ``var_floor``. The constrained MLE of
    a Gaussian variance is max(sample variance, floor) because the profile
    log likelihood is unimodal in the variance, so the floored value is still
    an exact minimiser and subadditivity (needed by PELT) is preserved. That
    is why the cost is written as len*log(2*pi*v) + SSE/v rather than the
    usual shortcut len*(log(2*pi*v) + 1), which is only valid unfloored.
    """

    name = "normal"
    min_size = 2
    params_per_dim = 2

    def __init__(self, var_floor=None) -> None:
        super().__init__()
        self.var_floor = var_floor

    def fit(self, y) -> "NormalMeanVarCost":
        arr = _as_2d(y)
        self.n, self.d = arr.shape
        if self.var_floor is None:
            base = arr.var(axis=0)
            self.floor_ = np.maximum(1e-6 * base, 1e-12)
        else:
            self.floor_ = np.broadcast_to(np.asarray(self.var_floor, float), (self.d,))
        self._s1 = _prefix(arr)
        self._s2 = _prefix(arr * arr)
        return self

    def cost_pairs(self, starts, ends) -> np.ndarray:
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        length = (ends - starts)[:, None].astype(float)
        s1 = self._s1[ends] - self._s1[starts]
        s2 = self._s2[ends] - self._s2[starts]
        sse = np.maximum(s2 - s1 * s1 / length, 0.0)
        v = np.maximum(sse / length, self.floor_)
        return (length * (LOG_2PI + np.log(v)) + sse / v).sum(axis=1)


class PoissonCost(Cost):
    """Change in rate of a Poisson count process.

    cost = 2 * (S - S*log(S/len)), the log(y!) terms are segmentation
    invariant and dropped. A segment of all zeros costs 0.
    """

    name = "poisson"
    min_size = 1
    params_per_dim = 1

    def fit(self, y) -> "PoissonCost":
        arr = _as_2d(y)
        if np.any(arr < 0):
            raise ValueError("Poisson cost needs non negative counts")
        self.n, self.d = arr.shape
        self._s1 = _prefix(arr)
        return self

    def cost_pairs(self, starts, ends) -> np.ndarray:
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        length = (ends - starts)[:, None].astype(float)
        s = self._s1[ends] - self._s1[starts]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(s > 0, s * np.log(np.where(s > 0, s, 1.0) / length), 0.0)
        return (2.0 * (s - term)).sum(axis=1)


COSTS = {"l2": L2Cost, "normal": NormalMeanVarCost, "poisson": PoissonCost}


def make_cost(cost="l2", **kwargs) -> Cost:
    """Turn a name or an existing Cost instance into a Cost instance."""
    if isinstance(cost, Cost):
        return cost
    try:
        return COSTS[cost](**kwargs)
    except KeyError:
        raise ValueError(f"unknown cost {cost!r}; choose from {sorted(COSTS)}") from None
