import io
import itertools
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from changepoint_lab import hausdorff, precision_recall_f1, rand_index, resolve_penalty
from changepoint_lab.cli import main
from changepoint_lab.metrics import labels


def brute_matches(true, pred, margin):
    best = 0
    for k in range(min(len(true), len(pred)), 0, -1):
        for ts in itertools.combinations(true, k):
            for ps in itertools.permutations(pred, k):
                if all(abs(a - b) <= margin for a, b in zip(ts, ps)):
                    return k
    return best


class MetricTests(unittest.TestCase):
    def test_matching_is_maximum_cardinality(self):
        # the counterexample that defeats closest pair first matching
        self.assertEqual(precision_recall_f1([0, 3], [2, 5], margin=2), (1.0, 1.0, 1.0))
        rng = np.random.default_rng(0)
        for _ in range(300):
            true = sorted(set(rng.integers(0, 30, rng.integers(0, 6)).tolist()))
            pred = sorted(set(rng.integers(0, 30, rng.integers(0, 6)).tolist()))
            margin = int(rng.integers(0, 5))
            p, r, _ = precision_recall_f1(true, pred, margin)
            tp = brute_matches(true, pred, margin)
            if pred:
                self.assertAlmostEqual(p, tp / len(pred))
            if true:
                self.assertAlmostEqual(r, tp / len(true))

    def test_rand_index_matches_pairwise_definition(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            n = int(rng.integers(2, 40))
            a = sorted(set(rng.integers(1, n, rng.integers(0, 4)).tolist())) if n > 1 else []
            b = sorted(set(rng.integers(1, n, rng.integers(0, 4)).tolist())) if n > 1 else []
            la, lb = labels(a, n), labels(b, n)
            agree = sum((la[i] == la[j]) == (lb[i] == lb[j]) for i in range(n) for j in range(i + 1, n))
            self.assertAlmostEqual(rand_index(a, b, n), agree / (n * (n - 1) / 2))

    def test_hausdorff(self):
        self.assertEqual(hausdorff([10, 50], [12, 50]), 2)
        self.assertEqual(hausdorff([10], [10, 90]), 80)
        self.assertEqual(hausdorff([], []), 0)
        self.assertEqual(hausdorff([5], []), float("inf"))

    def test_penalties(self):
        self.assertAlmostEqual(resolve_penalty("bic", 100, 1), 2 * np.log(100))
        self.assertEqual(resolve_penalty("aic", 100, 2), 6)
        self.assertEqual(resolve_penalty(7, 100, 2), 7)
        with self.assertRaises(ValueError):
            resolve_penalty("xyz", 10, 1)
        with self.assertRaises(ValueError):
            resolve_penalty(-1, 10, 1)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(args))
        return code, buf.getvalue()

    def test_detect_online_crops_on_a_file(self):
        rng = np.random.default_rng(3)
        y = np.concatenate([rng.normal(0, 1, 150), rng.normal(5, 1, 150)])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.csv")
            with open(path, "w") as fh:
                fh.write("time,value\n")
                for i, v in enumerate(y):
                    fh.write(f"{i},{v}\n")
            code, out = self.run_cli("detect", path, "--column", "1")
            self.assertEqual(code, 0)
            cp = int(out.split("changepoints:")[1].split()[0])
            self.assertLessEqual(abs(cp - 150), 2)
            code, out = self.run_cli("online", path, "--column", "1", "--hazard", "100")
            self.assertEqual(code, 0)
            self.assertIn("changepoint at", out)
            code, out = self.run_cli("crops", path, "--column", "1", "--pen-min", "1", "--pen-max", "50")
            self.assertEqual(code, 0)
            self.assertIn("distinct optimal segmentations", out)

    def test_demo_runs(self):
        code, out = self.run_cli("demo", "--n", "300", "--k", "3")
        self.assertEqual(code, 0)
        self.assertIn("pelt", out)


if __name__ == "__main__":
    unittest.main()
