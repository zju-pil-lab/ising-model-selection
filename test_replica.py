"""Independent numerical checks of the equations and local optimizers."""

import itertools
import unittest
import numpy as np
from scipy.optimize import minimize, brentq
from replica import Spectrum, SETTINGS, LocalLinear, LocalLogistic, entropy_state, solve, metrics


class ReplicaChecks(unittest.TestCase):
    def test_covariance_spectral_moments(self):
        for s in SETTINGS.values():
            sp = Spectrum(s.d, s.coupling)
            t = np.tanh(s.coupling)
            self.assertAlmostEqual(sp.mass, 1, 11)
            self.assertAlmostEqual(float(sp.w @ sp.gamma), 1, 11)
            expected = 1 + s.d * t * t / (1 - (s.d - 1) * t * t)
            self.assertAlmostEqual(float(sp.w @ (sp.gamma**2)), expected, 10)

    def test_entropy_reduction_against_original_eq19(self):
        s = SETTINGS["rr"]
        sp = Spectrum(s.d, s.coupling)
        for H, E in [(0.14, 0.18), (0.3, 0.4), (1.0, 1.5)]:
            v = entropy_state(H, E, s, 40, sp)
            lam = -1 / v["Gamma"]
            m1 = float(sp.w @ (1 / (lam - sp.gamma)))
            m2 = float(sp.w @ (1 / (lam - sp.gamma) ** 2))
            eta, R, F = v["eta"], v["R"], v["F"]
            self.assertAlmostEqual(E * eta, -m1, 10)
            self.assertAlmostEqual(v["K"], E * lam + 1 / eta, 10)
            self.assertAlmostEqual(v["chi"], 1 / E + eta * lam, 10)
            self.assertAlmostEqual(v["Q"], F / E**2 + R * lam - (-E * R + F * eta) * eta / m2, 9)
            self.assertAlmostEqual(H, R / eta**2 + F * lam + (-E * R + F * eta) * E / m2, 9)

    def test_linear_local_against_enumeration(self):
        s = SETTINGS["rr"]
        local = LocalLinear(s, 40, 12, 341)
        chi, Q = 0.17, 0.04
        j, _, _ = local.evaluate(chi, Q)
        c = local.mean - np.sqrt(Q) * local.cross
        threshold = s.lam * (1 + chi)
        for i in range(len(j)):
            candidates = []
            for signs in itertools.product([-1, 0, 1], repeat=s.d):
                signs = np.array(signs)
                active = np.flatnonzero(signs)
                v = np.zeros(s.d)
                if len(active):
                    v[active] = np.linalg.solve(
                        local.gram[i][np.ix_(active, active)],
                        c[i, active] - threshold * signs[active],
                    )
                if np.any(v * signs < 0):
                    continue
                grad = local.gram[i] @ v - c[i]
                if np.any(np.abs(grad[signs == 0]) > threshold + 1e-10):
                    continue
                candidates.append(v)
            self.assertEqual(len(candidates), 1)
            np.testing.assert_allclose(j[i], candidates[0], atol=2e-11)

    def test_logistic_local_against_independent_scipy(self):
        s = SETTINGS["grid"]
        local = LocalLogistic(s, 45, 4, 561)
        chi, Q = 0.5, 0.1
        j, _, _, kkt, _ = local.evaluate(chi, Q)
        self.assertLess(kkt, 1e-8)
        for i in range(4):
            x, z = local.x[i], local.z[i]

            def fun(v):
                coef = v[: s.d] - v[s.d :]
                b = x @ coef + np.sqrt(Q) * z
                y = np.array(
                    [
                        brentq(
                            lambda u: u - bi - chi * (1 - np.tanh(u)), bi, bi + 2 * chi, xtol=1e-14
                        )
                        for bi in b
                    ]
                )
                r = 1 - np.tanh(y)
                val = np.mean(0.5 * chi * r * r + np.logaddexp(0, -2 * y)) + s.lam * v.sum()
                g = -x.T @ r / len(x)
                return val, np.r_[g + s.lam, -g + s.lam]

            fit = minimize(
                fun,
                np.zeros(2 * s.d),
                jac=True,
                bounds=[(0, None)] * (2 * s.d),
                method="L-BFGS-B",
                options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 500},
            )
            np.testing.assert_allclose(j[i], fit.x[: s.d] - fit.x[s.d :], atol=2e-7)

    def test_stein_total_derivative(self):
        # Finite-difference every Gaussian coordinate after reoptimizing J.
        s = SETTINGS["rr"]
        local = LocalLogistic(s, 20, 1, 777)
        chi, Q = 0.2, 0.07
        j, e, _, _, _ = local.evaluate(chi, Q)
        derivative = 0.0
        eps = 2e-5
        for mu in range(local.M):
            original = local.z[0, mu]
            r = []
            for sign in [-1, 1]:
                local.z[0, mu] = original + sign * eps
                jj, *_ = local.evaluate(chi, Q)
                b = local.x[0, mu] @ jj[0] + np.sqrt(Q) * local.z[0, mu]
                y = brentq(lambda u: u - b - chi * (1 - np.tanh(u)), b, b + 2 * chi)
                r.append(1 - np.tanh(y))
            local.z[0, mu] = original
            derivative += (r[1] - r[0]) / (2 * eps)
        self.assertAlmostEqual(e, -derivative / local.M / np.sqrt(Q), 7)

    def test_precision_binomial_average(self):
        s = SETTINGS["rr"]
        j = np.array([[0.1, 0, 0], [0.1, 0.1, 0], [0.1, 0.1, 0.1]])
        state = {"p": 0.003, "R": 0.1}
        v = metrics(j, state, s)
        rng = np.random.default_rng(920)
        fp = rng.binomial(s.N - 1 - s.d, state["p"], size=(3, 300000))
        tp = np.arange(1, 4)[:, None]
        self.assertLess(abs(np.mean(tp / (tp + fp)) - v["precision"]), 0.001)
        self.assertGreater(v["precision"], v["precision_ratio_of_means"])

    def test_spectral_quadrature_convergence(self):
        a = solve(SETTINGS["grid"], 45, mc=2000, seed=14, order=96)
        b = solve(SETTINGS["grid"], 45, mc=2000, seed=14, order=384)
        for k in ["rss", "precision", "recall", "E", "H", "Q"]:
            self.assertAlmostEqual(a[k], b[k], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
