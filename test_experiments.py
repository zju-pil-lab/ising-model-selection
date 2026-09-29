"""Independent validation of actual graph sampling and high-dimensional fits."""

import unittest
import numpy as np
from scipy.optimize import minimize
from experiments import (
    random_regular_graph,
    periodic_grid,
    sample,
    fit,
    load_native,
    evaluate,
    one_repeat,
)


def exact_ising(neighbors, signs, coupling):
    N = len(neighbors)
    states = ((np.arange(2**N)[:, None] >> np.arange(N)) & 1) * 2.0 - 1
    logw = np.zeros(len(states))
    for i in range(N):
        for j, sign in zip(neighbors[i], signs[i]):
            if i < j:
                logw += coupling * sign * states[:, i] * states[:, j]
    weights = np.exp(logw - logw.max())
    weights /= weights.sum()
    return states, weights


class ExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = load_native()

    def test_graph_ensemble_and_sign_symmetry(self):
        rng = np.random.default_rng(61)
        counts = {}
        for _ in range(1500):
            n, s, _ = random_regular_graph(4, 2, rng)
            edges = []
            for i in range(4):
                self.assertEqual(len(set(n[i])), 2)
                self.assertNotIn(i, n[i])
                for j, sign in zip(n[i], s[i]):
                    self.assertEqual(sign, s[j, np.where(n[j] == i)[0][0]])
                    if i < j:
                        edges.append((i, int(j)))
            key = tuple(sorted(edges))
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 3)
        for value in counts.values():
            self.assertLess(abs(value - 500), 90)

    def test_periodic_grid(self):
        n, s, _ = periodic_grid(15)
        self.assertEqual(n.shape, (225, 4))
        self.assertTrue(np.all(s == 1))
        self.assertEqual(set(n[0]), {1, 14, 15, 210})
        for i in range(225):
            for j in n[i]:
                self.assertIn(i, n[j])

    def test_sampler_against_exact_signed_graph_and_grid(self):
        rng = np.random.default_rng(71)
        for n, s, k in [(*random_regular_graph(8, 3, rng)[:2], 0.4), (*periodic_grid(3)[:2], 0.2)]:
            states, w = exact_ising(n, s, k)
            expected = (states * w[:, None]).T @ states
            data, _ = sample(n, s, k, 9182, burn=2000, gap=5, count=60000, native=self.native)
            data = data.astype(float)
            blocks = np.array_split(data, 100)
            means = np.array([b.T @ b / len(b) for b in blocks])
            estimate = means.mean(axis=0)
            se = means.std(axis=0, ddof=1) / 10
            offdiag = ~np.eye(len(n), dtype=bool)
            z = (estimate - expected)[offdiag] / se[offdiag]
            self.assertLess(float(np.max(np.abs(z))), 5.5)
            self.assertLess(float(np.max(abs(data.mean(axis=0)))), 0.018)

    def test_fits_against_independent_scipy(self):
        rng = np.random.default_rng(83)
        data = rng.choice(np.array([-1, 1], np.int8), size=(75, 31))
        center = 5
        lam = 0.08
        x = (np.delete(data, center, axis=1) * data[:, center, None]).astype(float)
        P = x.shape[1]
        for loss in ["lin", "log"]:
            coef, stats = fit(data, center, lam, loss, native=self.native)

            def fun(v):
                j = v[:P] - v[P:]
                margin = x @ j
                if loss == "lin":
                    value = 0.5 * np.mean((1 - margin) ** 2)
                    g = x.T @ (margin - 1) / len(x)
                else:
                    value = np.logaddexp(0, -2 * margin).mean()
                    g = x.T @ (np.tanh(margin) - 1) / len(x)
                return value + lam * v.sum(), np.r_[g + lam, -g + lam]

            fit2 = minimize(
                fun,
                np.zeros(2 * P),
                jac=True,
                bounds=[(0, None)] * (2 * P),
                method="L-BFGS-B",
                options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 3000, "maxls": 50},
            )
            self.assertLess(stats["kkt"], 3e-10)
            self.assertAlmostEqual(stats["objective"], fit2.fun, 10)
            np.testing.assert_allclose(np.delete(coef, center), fit2.x[:P] - fit2.x[P:], atol=2e-6)

    def test_gauge_invariance_and_numerical_support(self):
        rng = np.random.default_rng(94)
        n, s, _ = random_regular_graph(24, 3, rng)
        data, _ = sample(n, s, 0.4, 477, burn=1000, gap=10, count=70, native=self.native)
        gauge = rng.choice(np.array([-1, 1], np.int8), size=24)
        center = 3
        for loss in ["lin", "log"]:
            j, _ = fit(data, center, 0.2, loss, native=self.native)
            transformed, _ = fit(data * gauge, center, 0.2, loss, native=self.native)
            np.testing.assert_allclose(transformed, j * gauge * gauge[center], atol=1e-10)
            truth = np.zeros(24)
            truth[n[center]] = 0.4 * s[center]
            a = evaluate(j, truth)
            b = evaluate(transformed, truth * gauge * gauge[center])
            self.assertEqual(a, b)

    def test_reproducible_sampler(self):
        n, s, _ = periodic_grid(3)
        a, _ = sample(n, s, 0.2, 999, burn=5, gap=2, count=100, native=self.native)
        b, _ = sample(n, s, 0.2, 999, burn=5, gap=2, count=100, native=self.native)
        np.testing.assert_array_equal(a, b)

    def test_polishing_of_previously_stalled_realization(self):
        # This full-size RR realization used to stall above the target KKT
        # threshold despite 20000 coordinate sweeps. It must not be discarded.
        result = one_repeat("rr", 8, 20260915, native=self.native)
        self.assertEqual(len(result["rows"]), 16)
        self.assertLessEqual(max(r["kkt"] for r in result["rows"]), 2e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
