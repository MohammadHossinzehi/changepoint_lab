"""Synthetic piecewise stationary signals with known changepoints."""

from __future__ import annotations

import numpy as np


def random_changepoints(n: int, k: int, min_gap: int, rng) -> list:
    """k changepoints in (0, n) with every segment at least min_gap long."""
    slack = n - (k + 1) * min_gap
    if slack < 0:
        raise ValueError("cannot fit that many segments")
    cuts = np.sort(rng.integers(0, slack + 1, size=k))
    return [int(c + (i + 1) * min_gap) for i, c in enumerate(cuts)]


def piecewise(n=500, k=4, kind="mean", min_gap=30, jump=2.0, seed=None):
    """Return (signal, true_changepoints).

    kind:
      "mean"    Gaussian, unit noise, mean jumps of size ~jump
      "meanvar" Gaussian, both mean and standard deviation change
      "poisson" counts whose rate changes by a factor of ~(1 + jump)
    """
    rng = np.random.default_rng(seed)
    cps = random_changepoints(n, k, min_gap, rng)
    bounds = [0, *cps, n]
    y = np.empty(n)
    level, scale, rate = 0.0, 1.0, 5.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if kind == "mean":
            y[a:b] = level + rng.normal(size=b - a)
            level += rng.choice([-1, 1]) * jump * rng.uniform(0.75, 1.25)
        elif kind == "meanvar":
            y[a:b] = level + scale * rng.normal(size=b - a)
            level += rng.choice([-1, 1]) * jump * rng.uniform(0.5, 1.0)
            scale = float(np.clip(scale * np.exp(rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)), 0.2, 5))
        elif kind == "poisson":
            y[a:b] = rng.poisson(rate, size=b - a)
            f = 1 + jump * rng.uniform(0.75, 1.25)
            rate = float(np.clip(rate * (f if rng.random() < 0.5 else 1 / f), 0.5, 80))
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return y, cps
