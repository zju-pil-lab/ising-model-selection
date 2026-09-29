"""Real full-graph Ising experiments, not the low-dimensional replica surrogate."""

from __future__ import annotations
import ctypes
from dataclasses import dataclass
from pathlib import Path
import subprocess
import numpy as np

BASE = Path(__file__).parent
ALPHAS = np.array([0.2, 0.4, 0.8, 1, 2, 3, 4, 5])


@dataclass(frozen=True)
class ExperimentSetting:
    N: int
    d: int
    coupling: float
    lam: float


EXPERIMENT_SETTINGS = {
    "rr": ExperimentSetting(200, 3, 0.4, 0.3),
    "grid": ExperimentSetting(225, 4, 0.2, 0.15),
}


def random_regular_graph(N, d, rng):
    """Exactly uniform labeled simple d-regular graph via configuration rejection.

    Reject the ENTIRE pairing if any self-loop or multiple edge is present.
    Local repairs would introduce a graph-ensemble bias and are not used.
    """
    if N * d % 2 or d >= N:
        raise ValueError("Invalid regular graph dimensions")
    stubs = np.repeat(np.arange(N, dtype=np.int32), d)
    tries = 0
    while True:
        tries += 1
        pairs = rng.permutation(stubs).reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue
        pairs.sort(axis=1)
        if len(np.unique(pairs[:, 0] * N + pairs[:, 1])) != len(pairs):
            continue
        neighbors = np.empty((N, d), np.int32)
        signs = np.empty((N, d), np.int8)
        counts = np.zeros(N, int)
        edge_signs = rng.choice(np.array([-1, 1], np.int8), size=len(pairs))
        for (i, j), s in zip(pairs, edge_signs):
            neighbors[i, counts[i]] = j
            signs[i, counts[i]] = s
            counts[i] += 1
            neighbors[j, counts[j]] = i
            signs[j, counts[j]] = s
            counts[j] += 1
        return neighbors, signs, tries


def periodic_grid(L=15):
    neighbors = np.empty((L * L, 4), np.int32)
    for i in range(L):
        for j in range(L):
            neighbors[i * L + j] = [
                ((i + 1) % L) * L + j,
                ((i - 1) % L) * L + j,
                i * L + (j + 1) % L,
                i * L + (j - 1) % L,
            ]
    return neighbors, np.ones_like(neighbors, dtype=np.int8), 1


def load_native():
    src = BASE / "experiment_native.cpp"
    lib = BASE / "experiment_native.so"
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            ["c++", "-O3", "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(lib)], check=True
        )
    native = ctypes.CDLL(str(lib))
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    native.sample_ising.argtypes = [
        i32,
        i8,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        i8,
        f64,
    ]
    native.sample_ising.restype = None
    native.fit_l1.argtypes = [
        i8,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        f64,
        f64,
    ]
    native.fit_l1.restype = ctypes.c_int
    return native


def sample(
    neighbors, signs, coupling, seed, burn=100000, gap=5, count=5000, initial=0, native=None
):
    native = native or load_native()
    N, d = neighbors.shape
    data = np.empty((count, N), np.int8)
    diag = np.zeros(2)
    native.sample_ising(
        np.ascontiguousarray(neighbors, dtype=np.int32),
        np.ascontiguousarray(signs, dtype=np.int8),
        N,
        d,
        coupling,
        seed,
        burn,
        gap,
        count,
        initial,
        data,
        diag,
    )
    return data, {"acceptance": float(diag[0]), "attempts": int(diag[1])}


def fit(data, center, lam, loss, tol=2e-10, native=None):
    native = native or load_native()
    M, N = data.shape
    indices = np.delete(np.arange(N), center)
    X = np.ascontiguousarray((data[:, indices] * data[:, center, None]).T)
    j = np.zeros(N - 1)
    stats = np.zeros(3)
    status = native.fit_l1(X, M, N - 1, lam, 0 if loss == "lin" else 1, tol, 2000, j, stats)
    # Tiny logistic objective improvements can hit the floating-point floor
    # in line searches. Solve the active smooth KKT equations directly, then
    # check EVERY inactive coordinate too. Do not drop stalled realizations.
    if status and loss == "log":
        from scipy.optimize import root

        design = X.T.astype(float)
        active = np.flatnonzero(j)
        for _ in range(20):
            signs = np.sign(j[active])
            xa = design[:, active]

            def equations(v):
                t = np.tanh(xa @ v)
                return xa.T @ (t - 1) / M + lam * signs, (xa.T * (1 - t * t)) @ xa / M

            solved = root(equations, j[active], jac=True, tol=1e-11)
            candidate = np.zeros_like(j)
            candidate[active] = solved.x
            flipped = candidate[active] * signs <= 0
            if np.any(flipped):
                candidate[active[flipped]] = 0
                j = candidate
                active = active[~flipped]
                continue
            j = candidate
            g = design.T @ (np.tanh(design @ j) - 1) / M
            violations = np.where(j != 0, np.abs(g + lam * np.sign(j)), np.maximum(abs(g) - lam, 0))
            if max(violations) <= tol:
                break
            add = np.flatnonzero((j == 0) & (abs(g) > lam + tol))
            if not len(add):
                break
            j[add] = -np.sign(g[add]) * 1e-12
            active = np.flatnonzero(j)
        margin = design @ j
        stats[0] = float(max(violations))
        stats[2] = float(np.logaddexp(0, -2 * margin).mean() + lam * abs(j).sum())
    if stats[0] > tol:
        raise RuntimeError(f"{loss} did not converge: KKT {stats[0]:.3g}, sweeps {stats[1]}")
    full = np.zeros(N)
    full[indices] = j
    return full, dict(kkt=float(stats[0]), sweeps=int(stats[1]), objective=float(stats[2]))


def evaluate(j, truth, zero=1e-8):
    selected = np.abs(j) > zero
    active = truth != 0
    tp = int(np.sum(selected & active))
    fp = int(np.sum(selected & ~active))
    d = int(active.sum())
    return dict(
        rss=float(np.sum((j - truth) ** 2)),
        tp=tp,
        fp=fp,
        recall=tp / d,
        precision=tp / (tp + fp) if tp + fp else 1.0,
        empty=int(tp + fp == 0),
        near_zero=int(np.sum((np.abs(j) > 1e-10) & (np.abs(j) < 1e-6))),
    )


def one_repeat(graph, repeat, seed, burn=100000, gap=5, pool_factor=5, alphas=ALPHAS, native=None):
    """Independent graph/chain per replicate, paired estimators and alpha points.

    Sharing a graph and pool between alpha points correlates points, but preserves
    1000 independent replicates WITHIN each point. Both fits use identical data.
    """
    native = native or load_native()
    s = EXPERIMENT_SETTINGS[graph]
    streams = np.random.SeedSequence([seed, 0 if graph == "rr" else 1, repeat]).spawn(4)
    rg, rm, rs, rc = [np.random.default_rng(x) for x in streams]
    neighbors, signs, tries = (
        random_regular_graph(s.N, s.d, rg) if graph == "rr" else periodic_grid()
    )
    Ms = np.rint(np.asarray(alphas) * s.N).astype(int)
    pool_size = int(pool_factor * max(Ms))
    mcmc_seed = int(rm.integers(0, 2**64, dtype=np.uint64))
    pool, sampling = sample(
        neighbors, signs, s.coupling, mcmc_seed, burn, gap, pool_size, native=native
    )
    rows = []
    coefficients = []
    centers = []
    for alpha, M in zip(alphas, Ms):
        subset = rs.choice(pool_size, size=M, replace=False)
        center = int(rc.integers(s.N))
        data = np.ascontiguousarray(pool[subset])
        truth = np.zeros(s.N)
        truth[neighbors[center]] = signs[center] * s.coupling
        centers.append(center)
        pair = []
        for loss in ["lin", "log"]:
            j, diagnostic = fit(data, center, s.lam, loss, native=native)
            pair.append(j)
            rows.append(
                dict(
                    graph=graph,
                    repeat=repeat,
                    seed=seed,
                    mcmc_seed=mcmc_seed,
                    alpha=float(alpha),
                    M=int(M),
                    center=center,
                    loss=loss,
                    **evaluate(j, truth),
                    **diagnostic,
                    **sampling,
                )
            )
        coefficients.append(pair)
    return dict(
        rows=rows,
        coefficients=np.array(coefficients),
        neighbors=neighbors,
        signs=signs,
        centers=np.array(centers),
        graph_pairing_attempts=tries,
        pool=pool if repeat == 0 else None,
    )
