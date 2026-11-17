import itertools
import math
import unittest

import numpy as np

from changepoint_lab import BOCPD, NormalGamma, PoissonGamma, bocpd, piecewise


def brute_force_run_length_posterior(x, model, h):
    """P(r_t = r | x_1:t) by enumerating every set of segment starts.

    Prior of a configuration with k starts among the t-1 possible positions
    is h^k (1-h)^(t-1-k); the likelihood is the product of the closed form
    segment evidences. Run length = t minus the last start.
    """
    t = len(x)
    post = np.zeros(t + 1)
    logs = []
    for mask in itertools.product([0, 1], repeat=t - 1):
        starts = [0] + [i + 1 for i, b in enumerate(mask) if b]
        k = len(starts) - 1
        lp = k * math.log(h) + (t - 1 - k) * math.log(1 - h)
        bounds = starts + [t]
        for a, b in zip(bounds[:-1], bounds[1:]):
            lp += model.log_marginal(x[a:b])
        logs.append((t - starts[-1], lp))
    m = max(v for _, v in logs)
    for r, v in logs:
        post[r] += math.exp(v - m)
    return post / post.sum()


class ExactPosterior(unittest.TestCase):
    def check(self, x, model, lam):
        det = BOCPD(model, hazard=lam, prune=0.0)
        for t in range(1, len(x) + 1):
            det.update(x[t - 1])
            rl, p = det.posterior()
            got = np.zeros(t + 1)
            got[rl] = p
            want = brute_force_run_length_posterior(x[:t], model, 1.0 / lam)
            np.testing.assert_allclose(got, want, atol=1e-10, err_msg=f"t={t}")

    def test_normal_gamma_matches_enumeration(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0, 1, 4), rng.normal(4, 0.5, 5)])
        self.check(x, NormalGamma(0.0, 0.5, 2.0, 1.5), lam=4.0)

    def test_poisson_gamma_matches_enumeration(self):
        x = np.array([1, 0, 2, 1, 9, 7, 11, 8, 10], dtype=float)
        self.check(x, PoissonGamma(2.0, 1.0), lam=5.0)

    def test_sequential_predictives_multiply_to_closed_form_evidence(self):
        rng = np.random.default_rng(3)
        x = rng.normal(2, 3, 12)
        for model in (NormalGamma(1.0, 0.3, 1.5, 2.0), PoissonGamma(3.0, 0.5)):
            data = np.round(np.abs(x)) if isinstance(model, PoissonGamma) else x
            state, total = model.init(), 0.0
            for v in data:
                total += float(model.log_pred(state, v)[0])
                state = model.update(state, v)
            self.assertAlmostEqual(total, model.log_marginal(data), places=9)

    def test_evidence_matches_enumeration(self):
        x = np.array([0.1, -0.4, 0.3, 3.9, 4.2, 4.0])
        model, lam = NormalGamma(0.0, 1.0, 1.0, 1.0), 3.0
        det = BOCPD(model, hazard=lam, prune=0.0)
        det.run(x)
        h, total = 1 / lam, []
        for mask in itertools.product([0, 1], repeat=len(x) - 1):
            starts = [0] + [i + 1 for i, b in enumerate(mask) if b]
            k = len(starts) - 1
            lp = k * math.log(h) + (len(x) - 1 - k) * math.log(1 - h)
            for a, b in zip(starts, starts[1:] + [len(x)]):
                lp += model.log_marginal(x[a:b])
            total.append(lp)
        m = max(total)
        self.assertAlmostEqual(det.log_evidence, m + math.log(sum(math.exp(v - m) for v in total)),
                               places=9)

    def test_new_run_probability_is_informative_in_this_convention(self):
        # With the "r = 0 means an empty run" convention P(r_t = 0) is
        # identically the hazard. Here x_t belongs to the new run, so
        # P(r_t = 1) carries evidence: tiny inside a segment, large at a jump.
        rng = np.random.default_rng(1)
        x = np.concatenate([np.zeros(20), np.full(20, 10.0)]) + rng.normal(0, .1, 40)
        det = BOCPD(NormalGamma(0, 1, 1, 1), hazard=50.0, prune=0.0)
        p_new = []
        for v in x:
            det.update(v)
            rl, p = det.posterior()
            p_new.append(float(p[rl == 1].sum()))
        self.assertGreater(p_new[20], 0.9)
        self.assertLess(max(p_new[5:20] + p_new[25:]), 0.02)


class Detection(unittest.TestCase):
    def test_detects_mean_changes_online(self):
        y, true = piecewise(n=600, k=4, kind="mean", jump=3.0, seed=4)
        res = bocpd(y, hazard=120)
        self.assertEqual(len(res.changepoints), len(true))
        for a, b in zip(res.changepoints, true):
            self.assertLessEqual(abs(a - b), 3)
        for c, t in res.detected_at.items():
            self.assertGreater(t, c)            # causal: declared after it happened
            self.assertLess(t - c, 40)          # and reasonably quickly

    def test_detects_rate_changes_in_counts(self):
        rng = np.random.default_rng(2)
        y = np.concatenate([rng.poisson(2, 150), rng.poisson(12, 150), rng.poisson(4, 150)]).astype(float)
        res = bocpd(y, model="poisson", hazard=150)
        self.assertEqual(len(res.changepoints), 2)
        self.assertLessEqual(abs(res.changepoints[0] - 150), 3)
        self.assertLessEqual(abs(res.changepoints[1] - 300), 5)

    def test_pruning_bounds_memory_without_changing_answer(self):
        y, _ = piecewise(n=1500, k=6, kind="mean", jump=3.0, seed=8)
        model = NormalGamma.from_data(y)
        exact = BOCPD(model, hazard=200, prune=0.0).run(y)
        fast = BOCPD(model, hazard=200, prune=1e-6).run(y)
        capped = BOCPD(model, hazard=200, prune=0.0, max_runs=100).run(y)
        self.assertEqual(exact.changepoints, fast.changepoints)
        self.assertEqual(exact.changepoints, capped.changepoints)
        self.assertEqual(exact.max_runs_kept, len(y))
        self.assertLess(fast.max_runs_kept, len(y) // 2)
        self.assertLessEqual(capped.max_runs_kept, 100)

    def test_no_false_alarms_on_stationary_noise(self):
        y = np.random.default_rng(6).normal(size=2000)
        self.assertEqual(bocpd(y, hazard=500).changepoints, [])

    def test_variable_hazard_callable(self):
        y, true = piecewise(n=300, k=2, kind="mean", jump=4.0, seed=1)
        # increasing hazard: long runs become more likely to end
        det = BOCPD(NormalGamma.from_data(y), hazard=lambda r: np.minimum(0.001 + r / 1e5, 0.5))
        res = det.run(y)
        self.assertEqual(len(res.changepoints), 2)


if __name__ == "__main__":
    unittest.main()
