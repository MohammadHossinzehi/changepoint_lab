"""Scaling and accuracy benchmark.

    python examples/benchmark.py

Part 1 times Optimal Partitioning against PELT as n grows with a fixed
average segment length (the regime where PELT's pruning pays off).
Part 2 averages detection accuracy over many random signals per method.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from changepoint_lab import (binary_segmentation, bocpd, optimal_partitioning, pelt,  # noqa: E402
                             piecewise, score)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def scaling():
    print("n       OP evals     PELT evals  ratio   OP time  PELT time  same answer")
    for n in (1000, 2000, 4000, 8000, 16000):
        y, _ = piecewise(n=n, k=n // 100, kind="mean", min_gap=40, seed=n)
        pe, tp = timed(lambda: pelt(y))
        if n <= 8000:
            op, to = timed(lambda: optimal_partitioning(y))
            ev = op.stats["cost_evaluations"]
            print(f"{n:<7} {ev:>9} {pe.stats['cost_evaluations']:>13} {ev / pe.stats['cost_evaluations']:6.0f}x"
                  f" {to:8.2f}s {tp:9.2f}s  {op.changepoints == pe.changepoints}")
        else:
            print(f"{n:<7} {'skipped':>9} {pe.stats['cost_evaluations']:>13} {'':>7} {'':>9} {tp:9.2f}s")


def accuracy(trials=30):
    print("\nkind     method                 F1(margin 5)  rand index  mean ms")
    for kind, cost, model in (("mean", "l2", "normal"), ("meanvar", "normal", "normal"),
                              ("poisson", "poisson", "poisson")):
        methods = {
            "pelt": lambda y: pelt(y, cost=cost).changepoints,
            "binary_segmentation": lambda y: binary_segmentation(y, cost=cost).changepoints,
            "bocpd (online)": lambda y: bocpd(y, model=model, hazard=100).changepoints,
        }
        for name, fn in methods.items():
            f1, ri, ms = [], [], []
            for seed in range(trials):
                y, true = piecewise(n=600, k=5, kind=kind, jump=1.5, seed=1000 + seed)
                cps, dt = timed(lambda: fn(y))
                s = score(true, cps, len(y), margin=5)
                f1.append(s["f1"])
                ri.append(s["rand_index"])
                ms.append(dt * 1000)
            print(f"{kind:<8} {name:<22} {np.mean(f1):12.3f} {np.mean(ri):11.4f} {np.mean(ms):8.1f}")


if __name__ == "__main__":
    scaling()
    accuracy()
