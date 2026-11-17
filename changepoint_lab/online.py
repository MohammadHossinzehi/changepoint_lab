"""Bayesian Online Changepoint Detection (Adams and MacKay 2007).

The model: the signal is a sequence of segments; at every step a new
segment starts with probability h(r) where r is the current run length;
inside a segment observations are iid given parameters drawn fresh from a
conjugate prior. BOCPD maintains the exact posterior over the current run
length r_t given x_1..x_t, one observation at a time, in O(#runs kept).

Run length convention used here (stated because papers disagree by one):
after seeing t observations, r_t is the number of observations in the
current segment, 1 <= r_t <= t. A changepoint "at index s" means x[s] is the
first observation of a new segment, so the current segment start is t - r_t.

Recursion, for t >= 2:
  p(r_t = r + 1, x_1:t) = p(r_{t-1} = r, x_1:t-1) * (1 - h(r)) * pred(x_t | run)
  p(r_t = 1,     x_1:t) = sum_r p(r_{t-1} = r, x_1:t-1) * h(r) * pred(x_t | prior)

tests/test_online.py checks this against brute force enumeration of all
2^(t-1) segmentations, so the indexing is pinned down exactly.

Why this convention: in the common "r_t = 0 is a fresh empty run"
formulation, x_t is scored before the split, and P(r_t = 0 | x_1:t) under a
constant hazard is identically H, so thresholding it detects nothing. Here
x_t is scored by the new run, so P(r_t = 1 | x_1:t) is evidence bearing (a
test checks it spikes at a jump). For reporting, the detector tracks the
maximum a posteriori segment start t - r_t and declares a changepoint once a
new start has persisted for a few steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_LOG_PI = math.log(math.pi)


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(a - m))))


_lgamma = np.frompyfunc(math.lgamma, 1, 1)


def lgamma(x) -> np.ndarray:
    return np.asarray(_lgamma(np.asarray(x, dtype=float)), dtype=float)


class NormalGamma:
    """Gaussian observations, unknown mean and precision (Normal Gamma prior).

    Prior: precision ~ Gamma(alpha0, beta0), mean | precision ~
    N(mu0, 1 / (kappa0 * precision)). The predictive is a Student t with
    2*alpha degrees of freedom, location mu and squared scale
    beta * (kappa + 1) / (alpha * kappa).
    """

    def __init__(self, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0) -> None:
        if kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0:
            raise ValueError("kappa0, alpha0, beta0 must be positive")
        self.prior = tuple(float(v) for v in (mu0, kappa0, alpha0, beta0))

    @classmethod
    def from_data(cls, x, kappa0=0.1, alpha0=1.0):
        """Empirical Bayes style default: centre on the data, scale beta0 so
        the prior mean variance matches a robust noise estimate."""
        x = np.asarray(x, float)
        mu0 = float(np.median(x))
        d = np.diff(x)
        s = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2) if x.size > 2 else 1.0
        s = s if s > 0 else (float(x.std()) or 1.0)
        return cls(mu0, kappa0, alpha0, alpha0 * s * s)

    def init(self):
        mu, k, a, b = self.prior
        return (np.array([mu]), np.array([k]), np.array([a]), np.array([b]))

    @staticmethod
    def log_pred(state, x: float) -> np.ndarray:
        mu, k, a, b = state
        scale2 = b * (k + 1.0) / (a * k)
        nu = 2.0 * a
        z = (x - mu) ** 2 / (nu * scale2)
        return (lgamma(a + 0.5) - lgamma(a) - 0.5 * (np.log(nu * scale2) + _LOG_PI)
                - (a + 0.5) * np.log1p(z))

    @staticmethod
    def update(state, x: float):
        mu, k, a, b = state
        k1 = k + 1.0
        return ((k * mu + x) / k1, k1, a + 0.5, b + k * (x - mu) ** 2 / (2.0 * k1))

    @staticmethod
    def concat(s1, s2):
        return tuple(np.concatenate((p, q)) for p, q in zip(s1, s2))

    @staticmethod
    def take(state, idx):
        return tuple(p[idx] for p in state)

    def log_marginal(self, x) -> float:
        """Closed form log evidence of one segment (independent of log_pred,
        used by the tests as a second opinion)."""
        x = np.asarray(x, float)
        mu0, k0, a0, b0 = self.prior
        n = x.size
        xbar = x.mean()
        kn = k0 + n
        an = a0 + n / 2.0
        bn = b0 + 0.5 * ((x - xbar) ** 2).sum() + k0 * n * (xbar - mu0) ** 2 / (2.0 * kn)
        return (math.lgamma(an) - math.lgamma(a0) + a0 * math.log(b0) - an * math.log(bn)
                + 0.5 * math.log(k0 / kn) - 0.5 * n * math.log(2.0 * math.pi))


class PoissonGamma:
    """Count observations with a Gamma(a0, b0) prior on the rate.

    Predictive is negative binomial.
    """

    def __init__(self, a0=1.0, b0=1.0) -> None:
        if a0 <= 0 or b0 <= 0:
            raise ValueError("a0 and b0 must be positive")
        self.prior = (float(a0), float(b0))

    @classmethod
    def from_data(cls, x, strength=1.0):
        m = float(np.mean(x)) or 1.0
        return cls(m * strength, strength)

    def init(self):
        a, b = self.prior
        return (np.array([a]), np.array([b]))

    @staticmethod
    def log_pred(state, x: float) -> np.ndarray:
        a, b = state
        return (lgamma(a + x) - lgamma(a) - math.lgamma(x + 1.0)
                + a * np.log(b / (b + 1.0)) - x * np.log(b + 1.0))

    @staticmethod
    def update(state, x: float):
        a, b = state
        return (a + x, b + 1.0)

    concat = staticmethod(NormalGamma.concat)
    take = staticmethod(NormalGamma.take)

    def log_marginal(self, x) -> float:
        x = np.asarray(x, float)
        a0, b0 = self.prior
        n, s = x.size, x.sum()
        return (math.lgamma(a0 + s) - math.lgamma(a0) + a0 * math.log(b0)
                - (a0 + s) * math.log(b0 + n) - sum(math.lgamma(v + 1.0) for v in x))


def constant_hazard(expected_run_length: float):
    """h(r) = 1 / lambda: geometric segment lengths with mean lambda."""
    if expected_run_length <= 1:
        raise ValueError("expected run length must exceed 1")
    h = 1.0 / expected_run_length
    return lambda r: np.full(np.shape(r), h)


@dataclass
class BOCPDResult:
    changepoints: list
    detected_at: dict           # changepoint -> time step when it was declared
    map_run_length: np.ndarray  # MAP r_t after each observation
    mean_run_length: np.ndarray
    log_evidence: float
    max_runs_kept: int
    posterior: list = field(default_factory=list)  # optional (run_lengths, probs) per t


class BOCPD:
    """Streaming detector. Call ``update(x)`` per observation, or ``run(xs)``.

    hazard: a number lambda (expected segment length) or a callable h(r).
    prune: drop run lengths whose posterior falls below this probability.
    max_runs: hard cap on hypotheses kept (most probable kept).
    persistence: steps a new MAP segment start must survive to be declared.
    """

    def __init__(self, model, hazard=250.0, prune=1e-8, max_runs=None, persistence=5):
        self.model = model
        self.hazard = constant_hazard(hazard) if not callable(hazard) else hazard
        self.prune = float(prune)
        self.max_runs = max_runs
        self.persistence = int(persistence)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.run_lengths = np.zeros(0, dtype=np.int64)
        self.log_post = np.zeros(0)
        self.state = None
        self.log_evidence = 0.0
        self._cand, self._cand_count = None, 0
        self.changepoints, self.detected_at = [], {}
        self.max_runs_kept = 0

    def update(self, x: float) -> dict:
        x = float(x)
        model = self.model
        prior = model.init()
        lp_prior = float(model.log_pred(prior, x)[0])
        if self.t == 0:
            log_joint = np.array([lp_prior])
            rl = np.array([1], dtype=np.int64)
            state = model.update(prior, x)
        else:
            h = np.clip(self.hazard(self.run_lengths), 1e-300, 1.0)
            lp = model.log_pred(self.state, x)
            with np.errstate(divide="ignore"):
                growth = self.log_post + np.log1p(-h) + lp
            cp = _logsumexp(self.log_post + np.log(h)) + lp_prior
            log_joint = np.concatenate(([cp], growth))
            rl = np.concatenate(([1], self.run_lengths + 1))
            state = model.concat(model.update(prior, x), model.update(self.state, x))
        z = _logsumexp(log_joint)
        self.log_evidence += z
        log_post = log_joint - z
        keep = np.flatnonzero(log_post >= math.log(self.prune)) if self.prune > 0 else np.arange(rl.size)
        if self.max_runs is not None and keep.size > self.max_runs:
            keep = keep[np.argsort(log_post[keep])[::-1][: self.max_runs]]
            keep.sort()
        if keep.size == 0:
            keep = np.array([int(np.argmax(log_post))])
        log_post = log_post[keep]
        log_post -= _logsumexp(log_post)
        self.run_lengths, self.log_post = rl[keep], log_post
        self.state = model.take(state, keep)
        self.t += 1
        self.max_runs_kept = max(self.max_runs_kept, keep.size)

        probs = np.exp(self.log_post)
        r_map = int(self.run_lengths[int(np.argmax(self.log_post))])
        start = self.t - r_map
        declared = None
        if start > 0 and (not self.changepoints or start > self.changepoints[-1]):
            if start == self._cand:
                self._cand_count += 1
            else:
                self._cand, self._cand_count = start, 1
            if self._cand_count >= self.persistence:
                self.changepoints.append(start)
                self.detected_at[start] = self.t
                declared = start
                self._cand, self._cand_count = None, 0
        else:
            self._cand, self._cand_count = None, 0
        return {"t": self.t, "map_run_length": r_map,
                "mean_run_length": float(probs @ self.run_lengths),
                "declared": declared}

    def posterior(self):
        """(run_lengths, probabilities) after the latest observation."""
        return self.run_lengths.copy(), np.exp(self.log_post)

    def run(self, xs, keep_posterior=False) -> BOCPDResult:
        self.reset()
        xs = np.asarray(xs, float).ravel()
        maps = np.zeros(xs.size, dtype=np.int64)
        means = np.zeros(xs.size)
        post = []
        for i, x in enumerate(xs):
            info = self.update(x)
            maps[i], means[i] = info["map_run_length"], info["mean_run_length"]
            if keep_posterior:
                post.append(self.posterior())
        return BOCPDResult(list(self.changepoints), dict(self.detected_at), maps, means,
                           self.log_evidence, self.max_runs_kept, post)


def bocpd(x, model="normal", hazard=250.0, **kwargs) -> BOCPDResult:
    """Convenience wrapper: pick a model by name with data driven priors."""
    if model == "normal":
        model = NormalGamma.from_data(x)
    elif model == "poisson":
        model = PoissonGamma.from_data(x)
    return BOCPD(model, hazard=hazard, **kwargs).run(x)
