"""Figure 2: 1000 full-graph repeats per point, resumable and seed reproducible."""

import argparse
from output_paths import output_directory
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments import ALPHAS, BASE, EXPERIMENT_SETTINGS, load_native, one_repeat


def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["graph"], r["loss"], r["alpha"]), []).append(r)
    out = []
    for (graph, loss, alpha), rr in sorted(groups.items()):
        n = len(rr)
        a = dict(graph=graph, loss=loss, alpha=alpha, M=rr[0]["M"], repeats=n)
        for key in ["rss", "precision", "recall", "tp", "fp", "empty"]:
            x = np.array([r[key] for r in rr])
            a[key] = float(x.mean())
            a[key + "_se"] = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        tp = np.array([r["tp"] for r in rr])
        total = tp + np.array([r["fp"] for r in rr])
        ratio = tp.mean() / total.mean()
        a["precision_ratio_of_means"] = float(ratio)
        a["precision_ratio_of_means_se"] = (
            float(np.std((tp - ratio * total) / total.mean(), ddof=1) / np.sqrt(n))
            if n > 1
            else 0.0
        )
        a["max_kkt"] = max(r["kkt"] for r in rr)
        a["max_sweeps"] = max(r["sweeps"] for r in rr)
        a["near_zero_count"] = sum(r["near_zero"] for r in rr)
        out.append(a)
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--burn", type=int, default=100000)
    parser.add_argument(
        "--gap",
        type=int,
        default=5,
        help="MCS between sample-pool snapshots; each MCS is N random-site attempts",
    )
    parser.add_argument("--pool-factor", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=BASE / "outputs/experiments")
    parser.add_argument("--graphs", nargs="+", choices=["rr", "grid"], default=["rr", "grid"])
    args = parser.parse_args()
    args.out = output_directory(args.out, parser)
    if (
        args.repeats < 1
        or args.seed < 0
        or args.burn < 0
        or args.gap < 1
        or args.pool_factor < 1
        or args.workers < 1
    ):
        parser.error("Use positive repeats/gap/pool-factor/workers and nonnegative seed/burn.")
    if len(set(args.graphs)) != len(args.graphs):
        parser.error("Each graph type may appear only once.")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    config = dict(
        seed=args.seed,
        burn=args.burn,
        gap=args.gap,
        pool_factor=args.pool_factor,
        alphas=ALPHAS.tolist(),
        solver_target_kkt=2e-10,
        solver_acceptance_kkt=2e-10,
        solver_polishing="active KKT root with global inactive checks",
        settings={k: asdict(v) for k, v in EXPERIMENT_SETTINGS.items()},
        source_sha256={
            p: hashlib.sha256((BASE / p).read_bytes()).hexdigest()
            for p in ["experiments.py", "experiment_native.cpp"]
        },
    )
    cp = out / "config.json"
    if cp.exists():
        if json.loads(cp.read_text()) != config:
            raise ValueError(
                "Output directory has a different experiment configuration; choose a new --out"
            )
    else:
        cp.write_text(json.dumps(config, indent=2))
    native = load_native()
    rows = []
    todo = []
    completed = 0
    total = len(args.graphs) * args.repeats
    start = time.perf_counter()
    for graph in args.graphs:
        for repeat in range(args.repeats):
            path = checkpoints / f"{graph}_{repeat:04d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    rows.extend(json.loads(str(z["rows"])))
                completed += 1
            else:
                todo.append((graph, repeat, path))
    print(
        f"{completed}/{total} complete; starting {len(todo)} independent graph/chain repetitions with {args.workers} workers.",
        flush=True,
    )

    def work(job):
        graph, repeat, path = job
        result = one_repeat(
            graph, repeat, args.seed, args.burn, args.gap, args.pool_factor, native=native
        )
        data = dict(
            rows=json.dumps(result["rows"]),
            coefficients=result["coefficients"],
            neighbors=result["neighbors"],
            signs=result["signs"],
            centers=result["centers"],
            graph_pairing_attempts=result["graph_pairing_attempts"],
        )
        if result["pool"] is not None:
            data["sample_pool"] = result["pool"]
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **data)
        temp.replace(path)
        return result["rows"]

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(work, job) for job in todo]
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
                completed += 1
            except Exception as exc:
                failures.append(str(exc))
                print(f"FAILED (will need rerun): {exc}", flush=True)
                continue
            if completed % 25 == 0 or completed == total:
                summary = aggregate(rows)
                (out / "summary.json").write_text(json.dumps(summary, indent=2))
                print(
                    f"{completed}/{total} graph/chain repetitions; {len(rows)} fits; elapsed {time.perf_counter() - start:.1f}s; max KKT {max(r['kkt'] for r in rows):.2e}",
                    flush=True,
                )
    rows.sort(key=lambda r: (r["graph"], r["repeat"], r["alpha"], r["loss"]))
    if not rows:
        (out / "failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError("All repetitions failed; no summary was written. See failures.json.")
    summary = aggregate(rows)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "summary.csv", summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    meta = dict(
        repeats_per_point=args.repeats,
        graphs=args.graphs,
        points=len(summary),
        total_fits=len(rows),
        status="incomplete" if failures else "complete",
        completed_repeats=completed,
        wall_seconds_this_invocation=time.perf_counter() - start,
        workers=args.workers,
        loaded_repeats=total - len(todo),
        numerical_support_tolerance=1e-8,
        max_kkt=max(r["kkt"] for r in rows),
        comparison_note="Same graph and sample pool shared across alpha points; independent repeats within each point. LinR/LogR paired on identical datasets.",
    )
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError(
            f"{len(failures)} repetitions failed; results are incomplete. Rerun after resolving the error."
        )
    if (out / "failures.json").exists():
        history = out / "history"
        history.mkdir(exist_ok=True)
        (out / "failures.json").replace(history / f"failures_before_success_{time.time_ns()}.json")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
