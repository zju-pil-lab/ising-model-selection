"""Recompute all four theory series; no data are fitted to the published plot."""

import argparse
from output_paths import output_directory
import csv
import json
from pathlib import Path
import time
from replica import SETTINGS, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-linear", type=int, default=30000)
    parser.add_argument("--mc-logistic", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=202109)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "outputs/theory"
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4, 0.8, 1, 2, 3, 4, 5])
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--losses", nargs="+", choices=["lin", "log"], default=["lin", "log"])
    parser.add_argument("--order", type=int, default=192)
    args = parser.parse_args()
    args.out = output_directory(args.out, parser)
    if (
        min(args.mc_linear, args.mc_logistic) < 2
        or args.order < 8
        or any(a <= 0 for a in args.alphas)
    ):
        parser.error("Use at least 2 Monte Carlo samples, order >= 8, and positive alpha values.")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.time()
    for name in args.settings:
        setting = SETTINGS[name]
        for loss in args.losses:
            for alpha in args.alphas:
                M = round(setting.N * alpha)
                seed = args.seed + M + 10000 * (name == "grid")
                mc = args.mc_linear if loss == "lin" else args.mc_logistic
                result = solve(setting, M, loss, mc, seed, args.order)
                result["graph"] = name
                results.append(result)
                (args.out / "theory.json").write_text(json.dumps(results, indent=2))
                print(
                    f"{name:4} {loss:3} alpha={alpha:4.1f} RSS={result['rss']:.6f} "
                    f"P(Eq26)={result['precision']:.6f} P(ratio)={result['precision_ratio_of_means']:.6f} "
                    f"Recall={result['recall']:.6f} EOS={result['eos_residual']:.1e} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    columns = [
        "graph",
        "loss",
        "M",
        "alpha",
        "mc",
        "seed",
        "rss",
        "precision",
        "precision_ratio_of_means",
        "recall",
        "rss_mc_se",
        "precision_mc_se",
        "recall_mc_se",
        "tp_mean",
        "fp_mean",
        "empty_probability",
        "E",
        "F",
        "H",
        "K",
        "Q",
        "R",
        "chi",
        "eta",
        "Gamma",
        "tau",
        "p",
        "eos_residual",
        "kkt_residual",
        "evaluations",
    ]
    with (args.out / "theory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    metadata = dict(
        runtime_seconds=time.time() - start,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
