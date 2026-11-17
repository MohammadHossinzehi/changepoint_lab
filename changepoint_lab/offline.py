"""Offline (whole signal) changepoint search.

Exact methods
  optimal_partitioning  O(n^2) dynamic program, penalised objective
  pelt                  same optimum, pruned; close to O(n) when changes
                        keep arriving (Killick, Fearnhead and Eckley 2012)
  segment_neighbourhood exact best segmentation for every K up to K_max
  crops                 every optimal segmentation over a penalty interval
                        (Haynes, Eckley and Fearnhead 2017)

Approximate
  binary_segmentation   greedy top down splitting, O(n log n)

Changepoints are reported as the index where a new segment starts, so the
segments are y[0:c1], y[c1:c2], ..., y[cK:n].
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .costs import Cost, make_cost
from .penalties import resolve_penalty


@dataclass
class Segmentation:
    changepoints: list
    cost: float          # unpenalised total cost
    penalty: float       # per changepoint penalty used (nan if none)
    n: int
    method: str
    stats: dict = field(default_factory=dict)

    @property
    def penalised_cost(self) -> float:
        return self.cost + self.penalty * len(self.changepoints)

    @property
    def segments(self) -> list:
        b = [0, *self.changepoints, self.n]
        return list(zip(b[:-1], b[1:]))

    def __repr__(self) -> str:
        return (f"Segmentation(method={self.method!r}, changepoints={self.changepoints}, "
                f"cost={self.cost:.4f}, penalty={self.penalty:.4f})")


def _setup(y, cost, min_size):
    c = make_cost(cost)
    c.fit(y)
    m = c.min_size if min_size is None else max(int(min_size), c.min_size)
    if c.n < m:
        raise ValueError(f"signal of length {c.n} is shorter than min_size={m}")
    return c, m


def _backtrack(last: np.ndarray, n: int) -> list:
    cps = []
    t = n
    while t > 0:
        s = int(last[t])
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


def optimal_partitioning(y, cost="l2", penalty="bic", min_size=None) -> Segmentation:
    """Exact penalised segmentation by plain dynamic programming.

    F(t) = min over s of F(s) + C(s, t) + beta, with F(0) = -beta.
    Kept deliberately simple: it is the reference PELT is tested against.
    """
    c, m = _setup(y, cost, min_size)
    n = c.n
    beta = resolve_penalty(penalty, n, c.n_params)
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, dtype=np.int64)
    evaluations = 0
    for t in range(m, n + 1):
        starts = np.concatenate(([0], np.arange(m, t - m + 1)))
        vals = F[starts] + c.cost_many(starts, t) + beta
        evaluations += starts.size
        j = int(np.argmin(vals))
        F[t] = vals[j]
        last[t] = starts[j]
    cps = _backtrack(last, n)
    return Segmentation(cps, c.total(cps), beta, n, "optimal_partitioning",
                        {"cost_evaluations": evaluations})


def pelt(y, cost="l2", penalty="bic", min_size=None) -> Segmentation:
    """Pruned Exact Linear Time search. Returns the same optimum as
    ``optimal_partitioning``.

    Pruning: if F(s) + C(s, t) > F(t) then for any T, splitting at t beats
    starting the last segment at s, because C(s, T) >= C(s, t) + C(t, T).

    The min_size subtlety: that argument needs the segment [t, T) to be
    admissible, i.e. T >= t + min_size. For T closer than that, s may still
    be the best start. So a pruned candidate stays eligible until
    T = t_pruned + min_size, and only then is it dropped. Many textbook
    implementations drop it immediately, which is only exact for
    min_size == 1; the oracle tests in tests/test_offline.py catch that bug.
    """
    c, m = _setup(y, cost, min_size)
    n = c.n
    beta = resolve_penalty(penalty, n, c.n_params)
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, dtype=np.int64)
    # candidate buffers, compacted in place each step (no reallocation)
    active = np.zeros(n + 1, dtype=np.int64)
    pruned_at = np.full(n + 1, np.inf)
    size = 1
    evaluations = 0
    max_active = 1
    for t in range(m, n + 1):
        act, pr = active[:size], pruned_at[:size]
        keep = t < pr + m
        if not keep.all():
            size = int(keep.sum())
            active[:size], pruned_at[:size] = act[keep], pr[keep]
            act, pr = active[:size], pruned_at[:size]
        # active is sorted, so the ready candidates (s <= t - m) are a prefix
        r = int(np.searchsorted(act, t - m, side="right"))
        S = act[:r]
        base = F[S] + c.cost_many(S, t)
        evaluations += r
        vals = base + beta
        j = int(np.argmin(vals))
        F[t] = vals[j]
        last[t] = S[j]
        # prune: mark (not drop) dominated candidates the first time they lose
        dominated = (base > F[t]) & np.isinf(pr[:r])
        pr[:r][dominated] = t
        active[size], pruned_at[size] = t, np.inf
        size += 1
        max_active = max(max_active, size)
    cps = _backtrack(last, n)
    return Segmentation(cps, c.total(cps), beta, n, "pelt",
                        {"cost_evaluations": evaluations, "max_candidates": max_active})


def segment_neighbourhood(y, max_changepoints: int, cost="l2", min_size=None) -> list:
    """Exact best segmentation with exactly k changepoints, k = 0..K.

    D[k][t] = min over s of D[k-1][s] + C(s, t). Costs O(K n^2) but needs no
    penalty, which makes it the ground truth for CROPS and the elbow plot.
    Returns a list indexed by k; entries are None if k changepoints do not
    fit given min_size.
    """
    c, m = _setup(y, cost, min_size)
    n = c.n
    K = int(max_changepoints)
    if K < 0:
        raise ValueError("max_changepoints must be >= 0")
    D = np.full((K + 1, n + 1), np.inf)
    arg = np.zeros((K + 1, n + 1), dtype=np.int64)
    ts = np.arange(m, n + 1)
    D[0, ts] = c.cost_pairs(np.zeros_like(ts), ts)
    for k in range(1, K + 1):
        for t in range((k + 1) * m, n + 1):
            starts = np.arange(k * m, t - m + 1)
            vals = D[k - 1, starts] + c.cost_many(starts, t)
            j = int(np.argmin(vals))
            D[k, t] = vals[j]
            arg[k, t] = starts[j]
    out = []
    for k in range(K + 1):
        if not np.isfinite(D[k, n]):
            out.append(None)
            continue
        cps, t = [], n
        for kk in range(k, 0, -1):
            t = int(arg[kk, t])
            cps.append(t)
        cps = sorted(cps)
        out.append(Segmentation(cps, float(D[k, n]), float("nan"), n, "segment_neighbourhood"))
    return out


def binary_segmentation(y, cost="l2", penalty="bic", n_changepoints=None,
                        min_size=None) -> Segmentation:
    """Greedy top down search.

    Repeatedly applies the single best split among all current segments
    (a heap keyed on gain), stopping when the best gain no longer pays for
    the penalty, or after ``n_changepoints`` splits if that is given.
    Fast, but not exact: it can miss short segments sandwiched between
    larger ones, which the demo makes visible.
    """
    c, m = _setup(y, cost, min_size)
    n = c.n
    beta = resolve_penalty(penalty, n, c.n_params)

    def best_split(a, b):
        if b - a < 2 * m:
            return None
        s = np.arange(a + m, b - m + 1)
        left = c.cost_pairs(np.full(s.shape, a), s)
        right = c.cost_pairs(s, np.full(s.shape, b))
        gains = c.cost(a, b) - left - right
        j = int(np.argmax(gains))
        return float(gains[j]), int(s[j])

    heap = []

    def push(a, b):
        r = best_split(a, b)
        if r is not None:
            heapq.heappush(heap, (-r[0], r[1], a, b))

    push(0, n)
    cps = []
    while heap:
        if n_changepoints is not None and len(cps) >= n_changepoints:
            break
        neg_gain, s, a, b = heapq.heappop(heap)
        if n_changepoints is None and -neg_gain <= beta:
            break
        cps.append(s)
        push(a, s)
        push(s, b)
    cps.sort()
    return Segmentation(cps, c.total(cps), beta if n_changepoints is None else float("nan"),
                        n, "binary_segmentation")


@dataclass
class CropsResult:
    """Every distinct optimal segmentation for penalties in [pen_min, pen_max].

    ``solutions`` is sorted by penalty interval; each entry is
    (pen_low, pen_high, Segmentation) meaning the segmentation is optimal for
    every penalty in that interval.
    """

    solutions: list
    pelt_runs: int

    def by_size(self) -> dict:
        return {len(s.changepoints): s for _, _, s in self.solutions}


def crops(y, pen_min: float, pen_max: float, cost="l2", min_size=None,
          tol: float = 1e-9) -> CropsResult:
    """Changepoints for a Range Of PenaltieS.

    The optimal penalised cost as a function of the penalty is the lower
    envelope of lines Q_k + beta*k, one per segmentation size k. CROPS finds
    every breakpoint of that envelope with at most (#sizes) extra PELT runs:
    for an interval whose endpoints disagree by more than one changepoint it
    evaluates PELT where their two lines cross, which either uncovers a new
    size (recurse on both halves) or proves none exists in between.
    """
    if not 0 <= pen_min <= pen_max:
        raise ValueError("need 0 <= pen_min <= pen_max")
    cache = {}

    def run(beta):
        key = round(beta, 12)
        if key not in cache:
            cache[key] = pelt(y, cost=cost, penalty=float(beta), min_size=min_size)
        return cache[key]

    lo, hi = run(pen_min), run(pen_max)
    stack = [(pen_min, pen_max)]
    while stack:
        b0, b1 = stack.pop()
        s0, s1 = run(b0), run(b1)
        k0, k1 = len(s0.changepoints), len(s1.changepoints)
        if k0 > k1 + 1:
            b_int = (s1.cost - s0.cost) / (k0 - k1)
            if not (b0 + tol < b_int < b1 - tol):
                continue
            si = run(b_int)
            if len(si.changepoints) != k1:
                stack.append((b0, b_int))
                stack.append((b_int, b1))
    del lo, hi
    best = {}
    for beta, seg in cache.items():
        k = len(seg.changepoints)
        if k not in best or seg.cost < best[k].cost - 1e-9:
            best[k] = seg
    ks = sorted(best, reverse=True)  # many changepoints = small penalties
    # each size k is optimal between its crossings with neighbours
    solutions = []
    for i, k in enumerate(ks):
        seg = best[k]
        low = pen_min if i == 0 else (best[k].cost - best[ks[i - 1]].cost) / (ks[i - 1] - k)
        high = pen_max if i == len(ks) - 1 else (best[ks[i + 1]].cost - seg.cost) / (k - ks[i + 1])
        solutions.append((max(low, pen_min), min(high, pen_max), seg))
    return CropsResult(solutions, len(cache))


def brute_force(y, cost="l2", penalty="bic", min_size=None):
    """Enumerate every admissible segmentation (exponential, n <= ~16).

    Returns (best penalised Segmentation, {k: best unpenalised cost}). Used
    only as a test oracle.
    """
    from itertools import combinations

    c, m = _setup(y, cost, min_size)
    n = c.n
    beta = resolve_penalty(penalty, n, c.n_params)
    best, best_val, per_k = None, np.inf, {}
    inner = range(m, n - m + 1)
    for k in range(0, n // m):
        for combo in combinations(inner, k):
            b = (0, *combo, n)
            if any(b[i + 1] - b[i] < m for i in range(len(b) - 1)):
                continue
            q = c.total(combo)
            if q < per_k.get(k, np.inf):
                per_k[k] = q
            if q + beta * k < best_val:
                best_val, best = q + beta * k, list(combo)
    return Segmentation(best, c.total(best), beta, n, "brute_force"), per_k
