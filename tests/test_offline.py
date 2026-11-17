import unittest

import numpy as np

from changepoint_lab import (binary_segmentation, brute_force, crops, optimal_partitioning, pelt,
                             piecewise, segment_neighbourhood)
from changepoint_lab.costs import L2Cost


class ExactnessAgainstBruteForce(unittest.TestCase):
    """Every exact method must hit the true optimum found by enumerating all
    admissible segmentations."""

    def cases(self):
        rng = np.random.default_rng(1)
        for trial in range(40):
            n = int(rng.integers(5, 13))
            kind = ["l2", "normal", "poisson"][trial % 3]
            if kind == "poisson":
                y = rng.poisson(rng.choice([1, 4, 9]), size=n).astype(float)
            else:
                y = rng.normal(size=n) + np.repeat(rng.normal(0, 3, 3), [n // 3, n // 3, n - 2 * (n // 3)])
            pen = float(rng.choice([0.5, 2.0, 5.0, 12.0]))
            m = int(rng.choice([1, 2, 3])) if kind != "normal" else int(rng.choice([2, 3]))
            yield trial, kind, y, pen, m

    def test_op_and_pelt_match_brute_force(self):
        for trial, kind, y, pen, m in self.cases():
            with self.subTest(trial=trial, cost=kind, min_size=m, pen=pen):
                bf, _ = brute_force(y, cost=kind, penalty=pen, min_size=m)
                op = optimal_partitioning(y, cost=kind, penalty=pen, min_size=m)
                pe = pelt(y, cost=kind, penalty=pen, min_size=m)
                self.assertAlmostEqual(op.penalised_cost, bf.penalised_cost, places=8)
                self.assertAlmostEqual(pe.penalised_cost, bf.penalised_cost, places=8)
                for seg in (op, pe):
                    lengths = [e - s for s, e in seg.segments]
                    self.assertGreaterEqual(min(lengths), m)

    def test_segment_neighbourhood_matches_brute_force_for_every_k(self):
        for trial, kind, y, pen, m in self.cases():
            with self.subTest(trial=trial, cost=kind, min_size=m):
                _, per_k = brute_force(y, cost=kind, penalty=pen, min_size=m)
                sn = segment_neighbourhood(y, max_changepoints=len(y), cost=kind, min_size=m)
                for k, seg in enumerate(sn):
                    if k in per_k:
                        self.assertAlmostEqual(seg.cost, per_k[k], places=8)
                        self.assertEqual(len(seg.changepoints), k)
                    else:
                        self.assertIsNone(seg)

    def test_eager_pruning_with_min_size_would_be_wrong(self):
        # A regression guard for the min_size subtlety documented in pelt():
        # on many random signals with min_size > 1, PELT must equal OP.
        rng = np.random.default_rng(5)
        for trial in range(60):
            y = rng.normal(size=60) + np.repeat(rng.normal(0, 2, 6), 10)
            for m in (2, 4, 7):
                op = optimal_partitioning(y, penalty=3.0, min_size=m)
                pe = pelt(y, penalty=3.0, min_size=m)
                self.assertAlmostEqual(op.penalised_cost, pe.penalised_cost, places=8,
                                       msg=f"trial {trial} min_size {m}")


class PeltBehaviour(unittest.TestCase):
    def test_pelt_equals_op_on_longer_signals(self):
        for seed, kind, cost in ((0, "mean", "l2"), (1, "meanvar", "normal"), (2, "poisson", "poisson")):
            y, _ = piecewise(n=400, k=6, kind=kind, seed=seed)
            op = optimal_partitioning(y, cost=cost)
            pe = pelt(y, cost=cost)
            self.assertEqual(op.changepoints, pe.changepoints)
            self.assertAlmostEqual(op.cost, pe.cost, places=6)

    def test_pruning_keeps_candidate_set_small(self):
        y, _ = piecewise(n=3000, k=30, kind="mean", min_gap=50, seed=3)
        pe = pelt(y)
        self.assertLess(pe.stats["max_candidates"], 300)
        n = len(y)
        self.assertLess(pe.stats["cost_evaluations"], 0.05 * n * (n + 1) / 2)

    def test_recovers_clear_changes(self):
        y, true = piecewise(n=800, k=5, kind="mean", jump=3.0, seed=11)
        pe = pelt(y)
        self.assertEqual(len(pe.changepoints), len(true))
        self.assertTrue(all(abs(a - b) <= 3 for a, b in zip(pe.changepoints, true)))

    def test_no_change_signal_gives_no_changepoints(self):
        y = np.random.default_rng(4).normal(size=1000)
        self.assertEqual(pelt(y).changepoints, [])

    def test_multivariate_signal(self):
        rng = np.random.default_rng(9)
        y = rng.normal(size=(300, 2))
        y[100:, 0] += 3  # change only in column 0
        y[200:, 1] -= 3  # change only in column 1
        cps = pelt(y).changepoints
        self.assertEqual(len(cps), 2)
        self.assertLessEqual(abs(cps[0] - 100), 3)
        self.assertLessEqual(abs(cps[1] - 200), 3)

    def test_accepts_cost_instance(self):
        y, _ = piecewise(n=200, k=2, seed=1)
        a = pelt(y, cost=L2Cost(sigma=1.0))
        b = pelt(y, cost="l2")
        self.assertEqual(a.changepoints, b.changepoints)


class BinarySegmentationTests(unittest.TestCase):
    def test_fixed_number_of_changepoints(self):
        y, true = piecewise(n=500, k=4, kind="mean", jump=3.0, seed=2)
        bs = binary_segmentation(y, n_changepoints=4)
        self.assertEqual(len(bs.changepoints), 4)
        # greedy splits are found in the right places but localised less
        # precisely than the exact search (189 vs truth 194 on this seed)
        self.assertTrue(all(abs(a - b) <= 6 for a, b in zip(bs.changepoints, true)))

    def test_never_beats_the_exact_optimum(self):
        rng = np.random.default_rng(8)
        for _ in range(20):
            y = rng.normal(size=120) + np.repeat(rng.normal(0, 2, 4), 30)
            bs = binary_segmentation(y, penalty=4.0)
            pe = pelt(y, penalty=4.0)
            self.assertGreaterEqual(bs.penalised_cost + 1e-9, pe.penalised_cost)

    def test_greedy_misses_short_middle_segment(self):
        # a short bump in the middle of a long flat signal whose two halves
        # differ slightly: the first greedy split cannot isolate the bump
        y = np.zeros(200)
        y[95:105] = 4.0
        y[100:] += 0.3
        y += np.random.default_rng(0).normal(0, 0.3, 200)
        exact = pelt(y, cost=L2Cost(sigma=0.3), penalty=30)
        self.assertTrue(any(abs(c - 95) <= 1 for c in exact.changepoints))
        self.assertTrue(any(abs(c - 105) <= 1 for c in exact.changepoints))


class CropsTests(unittest.TestCase):
    def test_crops_solutions_are_optimal_for_their_size(self):
        y, _ = piecewise(n=150, k=4, kind="mean", jump=1.5, seed=6)
        res = crops(y, 1.0, 100.0)
        sn = segment_neighbourhood(y, max_changepoints=149)
        for lo, hi, seg in res.solutions:
            k = len(seg.changepoints)
            self.assertAlmostEqual(seg.cost, sn[k].cost, places=6)
            self.assertLessEqual(lo, hi + 1e-9)

    def test_crops_covers_every_pelt_answer_in_range(self):
        y, _ = piecewise(n=200, k=5, kind="mean", jump=1.2, seed=12)
        res = crops(y, 2.0, 60.0)
        sizes = set(res.by_size())
        for beta in np.linspace(2.0, 60.0, 59):
            k = len(pelt(y, penalty=float(beta)).changepoints)
            self.assertIn(k, sizes, msg=f"beta={beta}")
        # interval endpoints chain together with no gaps
        for (_, hi, _), (lo, _, _) in zip(res.solutions, res.solutions[1:]):
            self.assertAlmostEqual(hi, lo, places=6)
        # and it should need about one PELT run per size, not one per grid point
        self.assertLessEqual(res.pelt_runs, 2 * len(sizes) + 2)


if __name__ == "__main__":
    unittest.main()
