"""Full-graph experimental means +/- one SE alongside the replica predictions."""

import os
import argparse
from output_paths import output_directory
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from plot_results import AXES, YLIMS

COLORS = {"lin": "#e43b36", "log": "#235ce6"}
LABELS = {"lin": r"$\ell_1$-LinR", "log": r"$\ell_1$-LogR"}


def subset(rows, graph, loss):
    return sorted(
        [r for r in rows if r["graph"] == graph and r["loss"] == loss], key=lambda r: r["alpha"]
    )


def plot(theory, experiment, out, ratio=False):
    precision = "precision_ratio_of_means" if ratio else "precision"
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.1))
    for row, graph in enumerate(["rr", "grid"]):
        for col, (key, label) in enumerate(
            [("rss", "RSS"), (precision, "Precision"), ("recall", "Recall")]
        ):
            ax = axes[row, col]
            for loss in ["lin", "log"]:
                t = subset(theory, graph, loss)
                e = subset(experiment, graph, loss)
                ax.plot(
                    [r["alpha"] for r in t],
                    [r[key] for r in t],
                    color=COLORS[loss],
                    ls="--",
                    lw=1.8,
                    label=LABELS[loss] + " replica",
                )
                ax.errorbar(
                    [r["alpha"] for r in e],
                    [r[key] for r in e],
                    yerr=[r[key + "_se"] for r in e],
                    color=COLORS[loss],
                    fmt="o" if loss == "lin" else "s",
                    markersize=4,
                    capsize=3,
                    lw=1.1,
                    label=LABELS[loss] + " simulation",
                )
            low, high = YLIMS[3 * row + col]
            ax.set_xlim(0, 5.08)
            ax.set_ylim(low, 1.025 if col else high)
            ax.set_xticks(range(6))
            ax.set_xlabel(r"$\alpha=M/N$", fontsize=11)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(label, fontsize=13, weight="bold", pad=10)
            ax.grid(color="#e5e7eb", lw=0.7)
            for spine in ax.spines.values():
                spine.set_color("#9ca3af")
            if col == 0:
                ax.legend(fontsize=9, loc="upper right", framealpha=0.95, edgecolor="#d1d5db")
    n = experiment[0]["repeats"]
    title = "Figure 2: full-graph simulation and replica theory"
    low_mc = any(r["mc"] < (30000 if r["loss"] == "lin" else 8000) for r in theory)
    if n != 1000 or low_mc:
        title = f"Figure 2 preview ({n} experimental repeats; reduced MC if requested)"
    fig.suptitle(title, fontsize=18, weight="bold", y=0.993)
    fig.text(
        0.5, 0.934, r"RR: $N=200,\ d=3,\ J_{ij}=\pm0.4,\ \lambda=0.3$", ha="center", fontsize=12
    )
    fig.text(
        0.5,
        0.493,
        r"Periodic $15\times15$ grid: $N=225,\ d=4,\ J_{ij}=+0.2,\ \lambda=0.15$",
        ha="center",
        fontsize=12,
    )
    convention = (
        "Precision: ratio of means." if ratio else "Precision: mean of per-run ratios, Eq. (26)."
    )
    fig.text(
        0.5,
        0.012,
        f"{n:,} independent repeats per point; error bars = one standard error.  {convention}",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.072, right=0.982, bottom=0.09, top=0.875, wspace=0.25, hspace=0.65)
    name = "figure2_full_ratio_of_means" if ratio else "figure2_full_reproduction"
    fig.savefig(out / (name + ".png"), dpi=180, facecolor="white")
    fig.savefig(out / (name + ".svg"), facecolor="white")
    plt.close(fig)


def overlay(experiment, out, source):
    im = Image.open(source).convert("RGB")
    fig = plt.figure(figsize=(15, 8.4))
    ax = fig.add_axes([0.02, 0.035, 0.96, 0.90])
    ax.imshow(im, alpha=0.38)
    ax.axis("off")
    for panel, (left, top, right, bottom) in enumerate(AXES):
        row, col = divmod(panel, 3)
        graph = ["rr", "grid"][row]
        key = ["rss", "precision", "recall"][col]
        low, high = YLIMS[panel]
        for loss, color, marker in [("lin", "#09833c", "^"), ("log", "#a019a0", "x")]:
            e = subset(experiment, graph, loss)
            xx = [left + r["alpha"] / 5 * (right - left) for r in e]
            yy = [bottom - (r[key] - low) / (high - low) * (bottom - top) for r in e]
            ee = [r[key + "_se"] / (high - low) * (bottom - top) for r in e]
            ax.errorbar(xx, yy, yerr=ee, color=color, fmt=marker, markersize=5, capsize=2, lw=1)
    fig.suptitle(
        "Published Figure 2 (faded) + newly simulated experimental points",
        fontsize=16,
        weight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.945,
        "Green triangles: LinR   |   Purple crosses: LogR   |   Mean per-run precision; error bars = one SE",
        ha="center",
        fontsize=10,
    )
    fig.savefig(out / "figure2_experiment_overlay.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, default=base / "data/reference/theory")
    parser.add_argument("--experiments", type=Path, default=base / "data/reference/experiments")
    parser.add_argument("--out", type=Path, default=base / "outputs/figures")
    parser.add_argument(
        "--source", type=Path, help="Optional original Figure 2 raster (1806 x 896)."
    )
    args = parser.parse_args()
    args.out = output_directory(args.out, parser)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    theory = json.loads((args.theory / "theory.json").read_text())
    experiment = json.loads((args.experiments / "summary.json").read_text())
    expected = {
        (graph, loss, alpha)
        for graph in ["rr", "grid"]
        for loss in ["lin", "log"]
        for alpha in [0.2, 0.4, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0]
    }
    for records in [theory, experiment]:
        if len(records) != 32 or {(r["graph"], r["loss"], r["alpha"]) for r in records} != expected:
            parser.error("Plotting requires all 32 Figure 2 points.")
    if len({r["repeats"] for r in experiment}) != 1:
        parser.error("All experimental points must have the same repeat count.")
    plot(theory, experiment, out)
    plot(theory, experiment, out, ratio=True)
    if args.source:
        if Image.open(args.source).size != (1806, 896):
            parser.error("The source raster must have dimensions 1806 x 896.")
        overlay(experiment, out, args.source)


if __name__ == "__main__":
    main()
