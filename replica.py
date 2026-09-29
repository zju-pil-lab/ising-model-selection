"""Finite-size replica theory of Meng, Obuchi & Kabashima, NeurIPS 2021.

Implements main-text (19), (21), (25), (26), supplement (67), (100), (109).
Only local d-dimensional surrogate estimators are solved.
"""

from __future__ import annotations
import ctypes
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import numpy as np
from scipy.optimize import brentq, root
from scipy.special import erfc, roots_legendre
from scipy.stats import binom


@dataclass(frozen=True)
class Setting:
    name: str
    N: int
    d: int
    coupling: float
    lam: float


SETTINGS = {
    "rr": Setting("RR", 200, 3, 0.4, 0.3),
    "grid": Setting("Grid (tree approximation)", 225, 4, 0.2, 0.15),
}


class Spectrum:
    """Eq. (109), integrated in adjacency angle to avoid edge singularities."""

    def __init__(self, d: int, coupling: float, order: int = 192):
        nodes, w = roots_legendre(order)
        theta = (nodes + 1) * np.pi / 2
        adjacency = 2 * np.sqrt(d - 1) * np.cos(theta)
        self.w = (
            w
            * np.pi
            / 2
            * d
            / (2 * np.pi)
            * 4
            * (d - 1)
            * np.sin(theta) ** 2
            / (d * d - adjacency**2)
        )
        t = np.tanh(coupling)
        self.gamma = (1 - t * t) / (1 + (d - 1) * t * t - t * adjacency)
        self.mass = float(self.w.sum())
        self.w /= self.mass

    def coefficients(self, p: float):
        """Return Gamma, K/E, B, C, V without subtracting 1/eta terms.

        p=E chi=K eta=erfc(tau/sqrt(2)), Gamma=-1/Lambda.
        Q=R B+(F/E^2) C; H=F B+R E^2 V.
        """
        g, w = self.gamma, self.w
        mean = float(w @ g)
        if p < 1e-10:
            gamma = p / mean
        else:

            def eq(log_g):
                v = np.exp(log_g) * g
                return float(w @ (v / (1 + v))) - p

            gamma = np.exp(
                brentq(eq, np.log(p / mean) - 5, np.log(p / mean / (1 - p)) + 5, xtol=1e-13)
            )
        f = 1 / (1 + gamma * g)
        A = float(w @ f)
        D = float(w @ (f * f))
        gf = g * f
        k = float(w @ gf) / A
        B = float(w @ (g * f * f)) / D
        variance = float(w @ ((gf - float(w @ gf)) ** 2))
        C = gamma * gamma * variance / D
        V = variance / (A * A * D)
        return gamma, k, B, C, V


def soft(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


class LocalLinear:
    """Exact sufficient-statistic sampling of Eq. (25); cost independent of M.

    Gauge x_j=s0*s_j makes the d local spins independent with E[x_j]=tanh K0.
    Conditional on the multinomial spin counts, each sum of Gaussian z is
    exactly N(0,count). This preserves the finite-M surrogate distribution.
    """

    def __init__(self, setting: Setting, M: int, mc: int, seed: int):
        d = setting.d
        self.setting, self.M, self.mc = setting, M, mc
        rng = np.random.default_rng(seed)
        self.states = ((np.arange(2**d)[:, None] >> np.arange(d)) & 1) * 2.0 - 1
        t = np.tanh(setting.coupling)
        probabilities = np.prod((1 + t * self.states) / 2, axis=1)
        counts = rng.multinomial(M, probabilities, size=mc)
        self.mean = counts @ self.states / M
        self.gram = np.einsum("ts,si,sj->tij", counts, self.states, self.states) / M
        self.cross = (np.sqrt(counts) * rng.standard_normal(counts.shape)) @ self.states / M

    def evaluate(self, chi: float, Q: float):
        c = self.mean - np.sqrt(Q) * self.cross
        j = np.zeros_like(c)
        threshold = self.setting.lam * (1 + chi)
        for sweep in range(500):
            old = j.copy()
            for k in range(j.shape[1]):
                rho = c[:, k] - np.einsum("ti,ti->t", self.gram[:, k, :], j) + j[:, k]
                j[:, k] = soft(rho, threshold)
            if np.max(np.abs(j - old)) < 1e-12 * (1 + np.max(np.abs(j))):
                break
        else:
            raise RuntimeError("Local linear coordinate descent failed")
        delta = 1 - 2 * np.sum(self.mean * j, axis=1) + np.einsum("ti,tij,tj->t", j, self.gram, j)
        gradient = np.einsum("tij,tj->ti", self.gram, j) - c
        residual = np.where(
            j != 0,
            np.abs(gradient + threshold * np.sign(j)),
            np.maximum(np.abs(gradient) - threshold, 0),
        )
        return j, float(delta.mean()), float(residual.max() / (1 + chi))


class LocalLogistic:
    """Eq. (100) with raw finite-M star samples and exact scalar prox.

    E uses Stein's identity to integrate the Gaussian score in Algorithm 3,
    including the derivative of the optimized active coefficients. This is
    an expectation-preserving variance reduction, vital as Q approaches 0.
    """

    def __init__(self, setting: Setting, M: int, mc: int, seed: int):
        self.setting, self.M, self.mc = setting, M, mc
        rng = np.random.default_rng(seed)
        self.x = np.ascontiguousarray(
            np.where(
                rng.random((mc, M, setting.d)) < (1 + np.tanh(setting.coupling)) / 2, 1.0, -1.0
            )
        )
        self.z = np.ascontiguousarray(rng.standard_normal((mc, M)))
        base = Path(__file__).parent
        lib = base / "logistic.so"
        src = base / "logistic.cpp"
        if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(
                ["c++", "-O3", "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(lib)],
                check=True,
            )
        self.lib = ctypes.CDLL(str(lib))
        arr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.solve_batch.argtypes = [
            arr,
            arr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            arr,
            arr,
        ]
        self.lib.solve_batch.restype = None

    def evaluate(self, chi: float, Q: float):
        j = np.zeros((self.mc, self.setting.d))
        stats = np.zeros((self.mc, 5))
        self.lib.solve_batch(
            self.x, self.z, self.mc, self.M, self.setting.d, chi, Q, self.setting.lam, j, stats
        )
        return (
            j,
            float(stats[:, 0].mean()),
            float(stats[:, 1].mean()),
            float(stats[:, 3].max()),
            stats,
        )


def entropy_state(H: float, E: float, setting: Setting, M: int, spectrum: Spectrum):
    tau = setting.lam * M / np.sqrt(H * setting.N)
    p = float(erfc(tau / np.sqrt(2)))
    h2 = (1 + tau * tau) * p - np.sqrt(2 / np.pi) * tau * np.exp(-tau * tau / 2)
    if not (0 < p < 1 and h2 > 0):
        raise ValueError("Invalid soft-threshold moment")
    Gamma, k, B, C, V = spectrum.coefficients(p)
    K = E * k
    chi = p / E
    eta = p / K
    R = H / (K * K) * h2
    F = H * (1 - h2 * V / (k * k)) / B
    Q = R * B + F / (E * E) * C
    return dict(
        H=H, E=E, K=K, chi=chi, eta=eta, R=R, F=F, Q=Q, Gamma=Gamma, p=p, tau=tau, B=B, C=C, V=V
    )


def metrics(j, state, setting):
    tp = np.count_nonzero(np.abs(j) > 1e-9, axis=1)
    n = setting.N - 1 - setting.d
    p = state["p"]
    # Exact average over independent inactive Gaussian scores in Eq. (21).
    fp = np.arange(n + 1)
    weights = binom.pmf(fp, n, p)
    precision_by_tp = np.zeros(setting.d + 1)
    precision_by_tp[0] = weights[0]  # empty predicted neighborhood -> precision 1
    for k in range(1, setting.d + 1):
        precision_by_tp[k] = np.sum(weights * k / (k + fp))
    precision = precision_by_tp[tp]
    recall = tp / setting.d
    rss = np.sum((j - setting.coupling) ** 2, axis=1) + state["R"]

    def se(x):
        return float(np.std(x, ddof=1) / np.sqrt(len(x)))

    return dict(
        rss=float(rss.mean()),
        precision=float(precision.mean()),
        recall=float(recall.mean()),
        rss_mc_se=se(rss),
        precision_mc_se=se(precision),
        recall_mc_se=se(recall),
        tp_mean=float(tp.mean()),
        fp_mean=n * p,
        empty_probability=float(np.mean(tp == 0) * weights[0]),
        precision_ratio_of_means=float(tp.mean() / (tp.mean() + n * p)),
        tp_histogram=np.bincount(tp, minlength=setting.d + 1).tolist(),
    )


def solve(setting: Setting, M: int, loss="lin", mc=4000, seed=202109, order=192):
    alpha = M / setting.N
    spectrum = Spectrum(setting.d, setting.coupling, order)
    local = (
        LocalLinear(setting, M, mc, seed) if loss == "lin" else LocalLogistic(setting, M, mc, seed)
    )
    evaluations = 0
    if loss == "lin":
        # E=alpha-p eliminates chi exactly for quadratic loss.
        from scipy.special import erfcinv

        hmax = (setting.lam * M) ** 2 / (2 * setting.N * erfcinv(min(alpha * 0.99, 0.99)) ** 2)
        lo = np.log(max(alpha * 0.04, (setting.lam * M) ** 2 / (25**2 * setting.N)))
        hi = np.log(min(alpha * 5, hmax))

        def equation(logh, full=False):
            nonlocal evaluations
            H = np.exp(logh)
            p = float(erfc(setting.lam * M / np.sqrt(2 * H * setting.N)))
            state = entropy_state(H, alpha - p, setting, M, spectrum)
            j, delta, kkt = local.evaluate(state["chi"], state["Q"])
            evaluations += 1
            target = alpha * (delta + state["Q"]) / (1 + state["chi"]) ** 2
            res = np.log(state["F"] / target)
            return (state, j, kkt, res) if full else res

        # A moderately small lower H keeps erfc representable at high M.
        while equation(lo) > 0:
            lo -= 0.4
        while equation(hi) < 0:
            next_hi = min(hi + 0.3, np.log(hmax))
            if next_hi == hi:
                raise RuntimeError("Could not bracket linear EOS root")
            hi = next_hi
        root_h = brentq(equation, lo, hi, xtol=2e-11)
        state, j, kkt, residual = equation(root_h, True)
        residual = abs(residual)
    elif loss == "log":

        def equations(v, full=False):
            nonlocal evaluations
            E, H = np.exp(v)
            state = entropy_state(H, E, setting, M, spectrum)
            j, e, f, kkt, stats = local.evaluate(state["chi"], state["Q"])
            evaluations += 1
            residual = np.log([E / (alpha * e), state["F"] / (alpha * f)])
            return (state, j, kkt, residual, stats) if full else residual

        fit = root(equations, np.log([alpha * 0.8, alpha * 0.8]), tol=1e-8)
        state, j, kkt, residual, stats = equations(fit.x, True)
        residual = float(np.max(np.abs(residual)))
        if residual > 2e-6:
            raise RuntimeError(f"Logistic EOS did not converge: {fit.message}, residual={residual}")
    else:
        raise ValueError(loss)
    if kkt > 1e-7:
        raise RuntimeError(f"Local KKT residual {kkt}")
    result = dict(
        setting=asdict(setting),
        M=M,
        alpha=alpha,
        loss=loss,
        mc=mc,
        seed=seed,
        spectral_order=order,
        eos_residual=float(residual),
        kkt_residual=kkt,
        evaluations=evaluations,
        **metrics(j, state, setting),
        **{k: float(v) for k, v in state.items()},
    )
    return result
