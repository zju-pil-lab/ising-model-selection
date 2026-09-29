"""Audit every replicate, coefficients, diagnostics and sampling sensitivity."""

import hashlib
import argparse
from output_paths import output_directory
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from experiments import evaluate, EXPERIMENT_SETTINGS, ALPHAS
from run_experiments import aggregate, write_csv

BASE = Path(__file__).parent


def checkpoints(folder, repeats):
    rows = []
    support_disagreements = 0
    max_metric_error = 0
    max_kkt_recomputed = 0
    max_coef_center = 0
    sign_sum = 0
    sign_count = 0
    support_deltas = {}
    coefficient_archive = {}
    for graph in ["rr", "grid"]:
        setting = EXPERIMENT_SETTINGS[graph]
        allcoef = []
        allnbr = []
        allsigns = []
        allcenters = []
        for repeat in range(repeats):
            p = folder / "checkpoints" / f"{graph}_{repeat:04d}.npz"
            with np.load(p, allow_pickle=False) as z:
                rr = json.loads(str(z["rows"]))
                coef = z["coefficients"]
                nbr = z["neighbors"]
                sign = z["signs"]
                centers = z["centers"]
                if repeat == 0:
                    pool = z["sample_pool"]
            assert coef.shape == (8, 2, setting.N) and len(rr) == 16
            for i in range(setting.N):
                assert len(set(nbr[i])) == setting.d and i not in nbr[i]
                for j, sg in zip(nbr[i], sign[i]):
                    assert sg == sign[j, np.flatnonzero(nbr[j] == i)[0]]
            if graph == "grid":
                assert np.all(sign == 1)
            else:
                sign_sum += int(sign.sum())
                sign_count += sign.size
            for a, alpha in enumerate(ALPHAS):
                truth = np.zeros(setting.N)
                center = int(centers[a])
                truth[nbr[center]] = setting.coupling * sign[center]
                for loss_index, loss in enumerate(["lin", "log"]):
                    r = rr[a * 2 + loss_index]
                    j = coef[a, loss_index]
                    assert (
                        r["repeat"] == repeat
                        and r["graph"] == graph
                        and r["alpha"] == alpha
                        and r["loss"] == loss
                    )
                    assert (
                        r["M"] == round(alpha * setting.N)
                        and r["center"] == center
                        and r["seed"] == 20260915
                    )
                    assert r["kkt"] <= 2e-10
                    metrics = evaluate(j, truth)
                    for key in ["rss", "precision", "recall", "tp", "fp"]:
                        error = abs(metrics[key] - r[key])
                        max_metric_error = max(error, max_metric_error)
                        assert error < 1e-13
                    max_coef_center = max(max_coef_center, abs(j[center]))
                    assert j[center] == 0
                    support_disagreements += int(np.any((abs(j) > 1e-10) != (abs(j) > 1e-6)))
                    if np.any((abs(j) > 1e-10) != (abs(j) > 1e-6)):
                        for threshold in [1e-10, 1e-6]:
                            changed = evaluate(j, truth, zero=threshold)
                            for key in ["precision", "recall"]:
                                group = (graph, loss, float(alpha), threshold, key)
                                support_deltas[group] = (
                                    support_deltas.get(group, 0.0)
                                    + (changed[key] - metrics[key]) / repeats
                                )
            # Reconstruct the subsets and check gradients independently in the
            # original, UNGAUGED coordinates for the saved first sample pool.
            if repeat == 0:
                streams = np.random.SeedSequence([20260915, 0 if graph == "rr" else 1, 0]).spawn(4)
                subsets = np.random.default_rng(streams[2])
                for a, alpha in enumerate(ALPHAS):
                    M = round(alpha * setting.N)
                    idx = subsets.choice(len(pool), M, replace=False)
                    c = int(centers[a])
                    data = pool[idx].astype(float)
                    indices = np.delete(np.arange(setting.N), c)
                    X = data[:, indices]
                    y = data[:, c]
                    for loss_index, loss in enumerate(["lin", "log"]):
                        j = coef[a, loss_index, indices]
                        if loss == "lin":
                            g = X.T @ (X @ j - y) / M
                        else:
                            g = X.T @ (y * (np.tanh(y * (X @ j)) - 1)) / M
                        violation = np.where(
                            j != 0,
                            abs(g + setting.lam * np.sign(j)),
                            np.maximum(abs(g) - setting.lam, 0),
                        )
                        max_kkt_recomputed = max(max_kkt_recomputed, float(max(violation)))
                        assert max(violation) < 2.01e-10
            rows.extend(rr)
            allcoef.append(coef)
            allnbr.append(nbr)
            allsigns.append(sign)
            allcenters.append(centers)
        coefficient_archive[graph + "_coefficients"] = np.array(allcoef)
        coefficient_archive[graph + "_neighbors"] = np.array(allnbr)
        coefficient_archive[graph + "_signs"] = np.array(allsigns)
        coefficient_archive[graph + "_centers"] = np.array(allcenters)
    return (
        rows,
        dict(
            checkpoints=2 * repeats,
            fits=len(rows),
            support_disagreements_1e10_to_1e6=support_disagreements,
            max_mean_metric_change_with_support_threshold=max(
                map(abs, support_deltas.values()), default=0.0
            ),
            support_threshold_changes=[
                dict(graph=k[0], loss=k[1], alpha=k[2], threshold=k[3], metric=k[4], mean_change=v)
                for k, v in sorted(support_deltas.items())
                if v != 0.0
            ],
            max_metric_reconstruction_error=max_metric_error,
            max_independently_recomputed_kkt=max_kkt_recomputed,
            max_center_coefficient=max_coef_center,
            rr_mean_coupling_sign=sign_sum / sign_count,
        ),
        coefficient_archive,
    )


def thinning_sensitivity(main_rows, thin_rows):
    main = {(r["graph"], r["repeat"], r["alpha"], r["loss"]): r for r in main_rows}
    groups = {}
    for r in thin_rows:
        groups.setdefault((r["graph"], r["loss"], r["alpha"]), []).append(
            (main[(r["graph"], r["repeat"], r["alpha"], r["loss"])], r)
        )
    result = []
    for (graph, loss, alpha), pairs in sorted(groups.items()):
        for key in ["rss", "precision", "recall"]:
            delta = np.array([b[key] - a[key] for a, b in pairs])
            se = delta.std(ddof=1) / np.sqrt(len(delta))
            mean = delta.mean()
            result.append(
                dict(
                    graph=graph,
                    loss=loss,
                    alpha=alpha,
                    metric=key,
                    paired_repeats=len(delta),
                    gap20_minus_gap5=float(mean),
                    difference_se=float(se),
                    z_score=float(mean / se) if se else 0.0,
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=BASE / "data/reference/experiments")
    parser.add_argument("--diagnostics", type=Path, default=BASE / "data/reference/diagnostics")
    parser.add_argument("--theory", type=Path, default=BASE / "data/reference/theory")
    parser.add_argument("--out", type=Path, default=BASE / "outputs/validation")
    args = parser.parse_args()
    args.out = output_directory(args.out, parser)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    diagnostics = args.diagnostics
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "test_replica", "test_experiments", "-v"],
        cwd=BASE,
        capture_output=True,
        text=True,
    )
    (out / "validation_tests.txt").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    rows, checks, coef = checkpoints(args.experiments, 1000)
    thin, thin_checks, _ = checkpoints(diagnostics / "thinning20", 64)
    np.savez_compressed(out / "all_coefficients_and_graphs.npz", **coef)
    summary = aggregate(rows)
    assert len(summary) == 32 and all(r["repeats"] == 1000 for r in summary)
    assert len(rows) == 32000
    sensitivity = thinning_sensitivity(rows, thin)
    write_csv(out / "thinning_sensitivity.csv", sensitivity)
    (out / "thinning_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    theory = json.loads((args.theory / "theory.json").read_text())
    lookup = {(r["graph"], r["loss"], r["alpha"]): r for r in theory}
    comparison = []
    for e in summary:
        t = lookup[(e["graph"], e["loss"], e["alpha"])]
        for metric in ["rss", "precision", "recall", "precision_ratio_of_means"]:
            comparison.append(
                dict(
                    graph=e["graph"],
                    loss=e["loss"],
                    alpha=e["alpha"],
                    metric=metric,
                    experiment=e[metric],
                    experiment_se=e[metric + "_se"],
                    replica=t[metric],
                    experiment_minus_replica=e[metric] - t[metric],
                )
            )
    write_csv(out / "experiment_vs_replica.csv", comparison)
    sources = [
        "experiment_native.cpp",
        "experiments.py",
        "run_experiments.py",
        "test_experiments.py",
        "audit_sampling.py",
        "plot_experiments.py",
        "validate_experiments.py",
    ]
    result = dict(
        unit_tests=14,
        unit_tests_exit_code=0,
        main=checks,
        thinning=thin_checks,
        max_stored_kkt=max(r["kkt"] for r in rows),
        thinning_max_abs_z=max(abs(r["z_score"]) for r in sensitivity),
        thinning_comparisons=len(sensitivity),
        source_sha256={p: hashlib.sha256((BASE / p).read_bytes()).hexdigest() for p in sources},
        limitations=[
            "Finite Ising MCMC samples are approximately iid, not mathematically independent.",
            "Alpha points share graph/pool; repetitions within each point are independent.",
            "A 64-repeat paired thinning audit has limited power; no multiple-testing significance claim is made.",
            "Differences from replica theory include finite-size/loop approximations, not just simulation error.",
        ],
    )
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
