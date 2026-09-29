"""Public entry point: plots, numerical checks, and fresh Figure 2 calculations."""

import argparse
from output_paths import output_directory
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "data/reference"


def run(script, *args):
    subprocess.run(
        [sys.executable, "-u", str(BASE / script), *map(str, args)], cwd=BASE, check=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["plot", "check", "verify-reference", "theory", "experiments", "all"]
    )
    parser.add_argument("--output", type=Path, default=BASE / "outputs")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke calculation with reduced MC counts / 3 short chains; not a scientific reproduction.",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive.")
    if args.quick and args.mode not in ["theory", "experiments", "all"]:
        parser.error("--quick only applies to theory, experiments, or all.")
    output = output_directory(args.output, parser)
    if args.quick:
        output = output / "quick"
        print(
            "SMOKE TEST: reduced sample counts and burn-in. Do not use for paper comparisons.",
            flush=True,
        )
    if args.mode == "check":
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-v"], cwd=BASE, check=True)
        return
    if args.mode == "verify-reference":
        run("tools/verify_reference.py")
        run("validate_experiments.py", "--out", output / "validation")
        return
    theory = REFERENCE / "theory"
    experiments = REFERENCE / "experiments"
    if args.mode in ["theory", "all"]:
        theory = output / "theory"
        extra = ["--mc-linear", 2000, "--mc-logistic", 500] if args.quick else []
        run("run_figure2.py", "--out", theory, *extra)
    if args.mode in ["experiments", "all"]:
        experiments = output / "experiments"
        extra = ["--repeats", 3, "--burn", 1000] if args.quick else []
        run("run_experiments.py", "--out", experiments, "--workers", args.workers, *extra)
    run("plot_results.py", "--results", theory, "--out", output / "figures")
    run(
        "plot_experiments.py",
        "--theory",
        theory,
        "--experiments",
        experiments,
        "--out",
        output / "figures",
    )
    print(f"Figures written to {output / 'figures'}", flush=True)


if __name__ == "__main__":
    main()
