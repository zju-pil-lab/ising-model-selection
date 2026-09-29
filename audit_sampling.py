"""Independent initialization and serial-correlation checks at Figure 2 settings."""

import json
import argparse
from output_paths import output_directory
from pathlib import Path
import numpy as np
from experiments import sample, load_native, periodic_grid


def observables(data, nbr, sign):
    N, d = nbr.shape
    bond = np.zeros(len(data))
    for k in range(d):
        bond += (data * data[:, nbr[:, k]] * sign[:, k]).sum(axis=1) / (N * d)
    values = [bond, data.mean(axis=1)]
    names = ["signed_bond_mean", "magnetization"]
    for node in [0, N // 4, N // 2, N - 1]:
        values += [data[:, node], data[:, node] * data[:, nbr[node, 0]] * sign[node, 0]]
        names += [f"spin_{node}", f"bond_{node}"]
    return np.array(values, dtype=float).T, names


def diagnostics(traces):
    C, L, P = traces.shape
    split = np.concatenate([traces[:, : L // 2], traces[:, L // 2 :]], axis=0)
    n = split.shape[1]
    W = split.var(axis=1, ddof=1).mean(axis=0)
    B = n * split.mean(axis=1).var(axis=0, ddof=1)
    rhat = np.sqrt(((n - 1) / n * W + B / n) / W)
    acfs = []
    for lag in [1, 2, 5, 10]:
        centered = traces - traces.mean(axis=1, keepdims=True)
        acfs.append(
            (centered[:, :-lag] * centered[:, lag:]).mean(axis=(0, 1))
            / (centered * centered).mean(axis=(0, 1))
        )
    return dict(
        classical_split_rhat=rhat.tolist(),
        autocorrelation_lags=[1, 2, 5, 10],
        autocorrelations=np.array(acfs).tolist(),
        means=traces.mean(axis=(0, 1)).tolist(),
        chains=C,
        draws_per_chain=L,
    )


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=base / "data/reference/experiments")
    parser.add_argument("--out", type=Path, default=base / "outputs/diagnostics")
    args = parser.parse_args()
    args.out = output_directory(args.out, parser)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    native = load_native()
    report = {}
    for graph in ["rr", "grid"]:
        path = args.experiments / "checkpoints" / f"{graph}_0000.npz"
        if graph == "grid":
            nbr, sign, _ = periodic_grid()
        else:
            if not path.exists():
                raise RuntimeError(f"Wait until {path} exists")
            with np.load(path, allow_pickle=False) as z:
                nbr = z["neighbors"]
                sign = z["signs"]
        pool_traces = []
        selected_traces = []
        for chain, initial in enumerate([0, 1, -1, 0]):
            pool, _ = sample(
                nbr,
                sign,
                0.4 if graph == "rr" else 0.2,
                20269200 + chain,
                burn=100000,
                gap=5,
                count=10000,
                initial=initial,
                native=native,
            )
            obs, names = observables(pool, nbr, sign)
            pool_traces.append(obs)
            idx = np.sort(np.random.default_rng(8200 + chain).choice(10000, 2000, replace=False))
            selected_traces.append(obs[idx])
        report[graph] = dict(
            observables=names,
            initializations=["random", "all_plus", "all_minus", "random"],
            pool_gap5=diagnostics(np.array(pool_traces)),
            retained_random_one_fifth=diagnostics(np.array(selected_traces)),
        )
        print(graph, json.dumps(report[graph]), flush=True)
    (out / "sampling_audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
