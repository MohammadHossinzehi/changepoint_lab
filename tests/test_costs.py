import math
import unittest

import numpy as np

from changepoint_lab.costs import L2Cost, NormalMeanVarCost, PoissonCost, mad_sigma, make_cost


def naive_l2(y, a, b, sigma):
    seg = y[a:b]
    return float((((seg - seg.mean(axis=0)) ** 2).sum(axis=0) / sigma ** 2).sum())


def naive_normal(y, a, b, floor):
    seg = y[a:b]
    out = 0.0
    for j in range(seg.shape[1]):
        v = max(seg[:, j].var(), floor[j])
        sse = ((seg[:, j] - seg[:, j].mean()) ** 2).sum()
        out += len(seg) * math.log(2 * math.pi * v) + sse / v
    return out


def naive_poisson(y, a, b):
    seg = y[a:b, 0]
    lam = seg.mean()
    if lam == 0:
        return 0.0
    return 2 * (len(seg) * lam - seg.sum() * math.log(lam))


class CostTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_prefix_sum_costs_match_direct_formulas(self):
        y = self.rng.normal(size=(40, 3)) * [1, 3, 0.5] + [0, 10, -2]
        l2 = L2Cost(sigma=[1.0, 2.0, 0.5]).fit(y)
        nm = NormalMeanVarCost().fit(y)
        for a in range(0, 38, 3):
            for b in range(a + 2, 41, 5):
                self.assertAlmostEqual(l2.cost(a, b), naive_l2(y, a, b, l2.sigma_), places=8)
                self.assertAlmostEqual(nm.cost(a, b), naive_normal(y, a, b, nm.floor_), places=7)
        counts = self.rng.poisson(4, size=(50, 1)).astype(float)
        counts[10:20] = 0
        pc = PoissonCost().fit(counts)
        for a in range(0, 49, 4):
            for b in range(a + 1, 51, 7):
                self.assertAlmostEqual(pc.cost(a, b), naive_poisson(counts, a, b), places=8)

    def test_splitting_never_increases_cost(self):
        # the property PELT's pruning depends on, including floored variance
        y = np.concatenate([self.rng.normal(size=30), np.full(10, 3.0), self.rng.normal(5, 2, 30)])
        counts = self.rng.poisson(3, size=70).astype(float)
        for cost, data, m in ((L2Cost(), y, 1), (NormalMeanVarCost(), y, 2), (PoissonCost(), counts, 1)):
            cost.fit(data)
            for _ in range(400):
                a = int(self.rng.integers(0, 70 - 2 * m))
                b = int(self.rng.integers(a + 2 * m, 71))
                s = int(self.rng.integers(a + m, b - m + 1))
                self.assertGreaterEqual(cost.cost(a, b) + 1e-9, cost.cost(a, s) + cost.cost(s, b),
                                        msg=f"{cost.name} [{a},{s},{b})")

    def test_constant_segment_is_finite_under_variance_floor(self):
        c = NormalMeanVarCost().fit(np.ones(20))
        self.assertTrue(np.isfinite(c.cost(0, 20)))

    def test_total_and_vectorised_agree(self):
        y = self.rng.normal(size=60)
        c = L2Cost().fit(y)
        cps = [10, 25, 40]
        manual = c.cost(0, 10) + c.cost(10, 25) + c.cost(25, 40) + c.cost(40, 60)
        self.assertAlmostEqual(c.total(cps), manual)
        many = c.cost_many(np.array([0, 5, 30]), 50)
        for s, v in zip([0, 5, 30], many):
            self.assertAlmostEqual(v, c.cost(s, 50))

    def test_mad_sigma_is_robust_to_mean_shifts(self):
        y = np.concatenate([self.rng.normal(0, 2, 500), self.rng.normal(50, 2, 500)])
        self.assertAlmostEqual(float(mad_sigma(y)[0]), 2.0, delta=0.25)
        self.assertGreater(y.std(), 20)  # the naive estimate would be useless

    def test_validation(self):
        with self.assertRaises(ValueError):
            make_cost("nope")
        with self.assertRaises(ValueError):
            PoissonCost().fit([1, -1, 2])
        with self.assertRaises(ValueError):
            L2Cost().fit([1.0, float("nan")])
        with self.assertRaises(IndexError):
            L2Cost().fit([1.0, 2.0]).cost(1, 1)


if __name__ == "__main__":
    unittest.main()
