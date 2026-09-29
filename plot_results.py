"""Plot independently computed theory and compare with the source raster.

Source pixels enter comparison only, never the replica solver.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
import argparse
from output_paths import output_directory
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

COLORS = {"lin": "#e33b35", "log": "#235ce6"}
LABELS = {"lin": r"$\ell_1$-LinR", "log": r"$\ell_1$-LogR"}
AXES = [
    (235, 67, 620, 374),
    (742, 67, 1127, 374),
    (1249, 67, 1634, 374),
    (235, 492, 620, 797),
    (742, 492, 1127, 797),
    (1249, 492, 1634, 797),
]
YLIMS = [(0.30, 0.40), (0.2, 1), (0.4, 1), (0.05, 0.35), (0, 1), (0, 1)]
LEGENDS = [
    (320, 78, 612, 205),
    (820, 235, 1119, 357),
    (1334, 233, 1625, 360),
    (320, 505, 612, 632),
    (829, 657, 1119, 783),
    (1334, 652, 1625, 779),
]


def series(data, graph, loss):
    return sorted(
        [r for r in data if r["graph"] == graph and r["loss"] == loss], key=lambda r: r["alpha"]
    )


def figure(data, out, convention):
    pkey = "precision" if convention == "eq26" else "precision_ratio_of_means"
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.9))
    for row, graph in enumerate(["rr", "grid"]):
        for col, (key, label) in enumerate(
            [("rss", "RSS"), (pkey, "Precision"), ("recall", "Recall")]
        ):
            ax = axes[row, col]
            for loss in ["lin", "log"]:
                records = series(data, graph, loss)
                x = [r["alpha"] for r in records]
                y = [r[key] for r in records]
                ax.plot(
                    x,
                    y,
                    color=COLORS[loss],
                    linestyle="--" if loss == "lin" else (0, (5, 3)),
                    lw=2,
                    label=LABELS[loss],
                )
                ax.scatter(
                    x, y, color=COLORS[loss], s=13, marker="o" if loss == "lin" else "s", zorder=3
                )
            low, high = YLIMS[row * 3 + col]
            ax.set_xlim(0, 5.08)
            ax.set_ylim(low, 1.025 if col in (1, 2) else high)
            ax.set_xticks(range(6))
            ax.set_xlabel(r"$\alpha=M/N$", fontsize=11)
            ax.set_ylabel(label, fontsize=12)
            ax.grid(color="#e5e7eb", lw=0.7)
            for spine in ax.spines.values():
                spine.set_color("#9ca3af")
            ax.tick_params(labelsize=10)
            ax.set_title(label, fontsize=13, weight="bold", pad=10)
            if col == 0:
                ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#d1d5db")
    fig.suptitle("Replica theory: Figure 2 reproduction", fontsize=19, weight="bold", y=0.99)
    fig.text(0.5, 0.932, r"RR: $N=200,\ d=3,\ K_0=0.4,\ \lambda=0.3$", ha="center", fontsize=12)
    fig.text(
        0.5,
        0.493,
        r"Grid parameters with tree approximation: $N=225,\ d=4,\ K_0=0.2,\ \lambda=0.15$",
        ha="center",
        fontsize=12,
    )
    footer = (
        "Precision = mean of TP / (TP + FP), as written in Eq. (26)."
        if convention == "eq26"
        else "Precision = mean(TP) / [mean(TP) + mean(FP)]. Alternative aggregation; see README."
    )
    fig.text(0.5, 0.012, footer, ha="center", fontsize=10, color="#4b5563")
    fig.subplots_adjust(left=0.072, right=0.982, bottom=0.088, top=0.875, wspace=0.25, hspace=0.65)
    fig.savefig(out / f"figure2_{convention}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"figure2_{convention}.svg", facecolor="white")
    plt.close(fig)


def overlay(data, out, source):
    im = Image.open(source).convert("RGB")
    w, h = im.size
    fig = plt.figure(figsize=(15, 8.4))
    ax = fig.add_axes([0.02, 0.035, 0.96, 0.90])
    ax.imshow(im, alpha=0.47)
    ax.axis("off")
    for panel, (left, top, right, bottom) in enumerate(AXES):
        row, col = divmod(panel, 3)
        graph = ["rr", "grid"][row]
        key = ["rss", "precision_ratio_of_means", "recall"][col]
        low, high = YLIMS[panel]
        for loss, color, ls in [("lin", "#09833c", "-"), ("log", "#9b179c", (0, (2, 2)))]:
            records = series(data, graph, loss)
            xx = [left + r["alpha"] / 5 * (right - left) for r in records]
            yy = [bottom - (r[key] - low) / (high - low) * (bottom - top) for r in records]
            ax.plot(xx, yy, color=color, lw=1.55, ls=ls)
    fig.suptitle(
        "Published Figure 2 (faded) + recomputed replica theory",
        fontsize=16,
        weight="bold",
        y=0.988,
    )
    fig.text(
        0.5,
        0.944,
        "Green: LinR   |   Purple: LogR   |   Precision uses ratio of means for this comparison",
        ha="center",
        fontsize=10,
    )
    fig.savefig(out / "figure2_overlay.png", dpi=180, facecolor="white")
    plt.close(fig)


def raster_comparison(data, out, source):
    """Read dashed-line pixels away from experimental markers and legends.

    These are approximate raster measurements, not recovered numeric source data.
    The shortest alpha=.2 to .4 segment cannot be separated from markers, and is
    intentionally excluded. Line width/antialiasing remain an uncertainty.
    """
    rgb = np.asarray(Image.open(source).convert("RGB"))
    summaries = []
    measurements = []
    for panel, (left, top, right, bottom) in enumerate(AXES):
        row, col = divmod(panel, 3)
        graph = ["rr", "grid"][row]
        key = ["rss", "precision", "recall"][col]
        low, high = YLIMS[panel]
        lx, ly, rx, ry = LEGENDS[panel]
        for loss in ["lin", "log"]:
            records = series(data, graph, loss)
            knots = np.array([r["alpha"] for r in records])
            xs = left + knots / 5 * (right - left)
            channel = 0 if loss == "lin" else 2
            mask = (
                (rgb[:, :, channel] > 180)
                & (rgb[:, :, 1] < 100)
                & (rgb[:, :, 2 if loss == "lin" else 0] < 100)
            )
            points = []
            for x in range(left + 1, right):
                if min(abs(xs - x)) < 9:
                    continue
                if x < left + 0.4 / 5 * (right - left):
                    continue
                ys = np.flatnonzero(mask[top : bottom + 1, x]) + top
                if lx <= x <= rx:
                    ys = ys[(ys < ly) | (ys > ry)]
                if not len(ys):
                    continue
                # Curves are single valued after markers and legends are excluded.
                if np.ptp(ys) > 8:
                    continue
                alpha = (x - left) / (right - left) * 5
                observed = low + (bottom - float(np.median(ys))) / (bottom - top) * (high - low)
                computed = float(np.interp(alpha, knots, [r[key] for r in records]))
                alternate = float(
                    np.interp(
                        alpha,
                        knots,
                        [r["precision_ratio_of_means" if col == 1 else key] for r in records],
                    )
                )
                points.append((observed, computed, alternate))
                measurements.append(
                    dict(
                        graph=graph,
                        loss=loss,
                        metric=key,
                        alpha=alpha,
                        published_raster=observed,
                        recomputed_eq26=computed,
                        recomputed_ratio=alternate,
                    )
                )
            if points:
                p = np.array(points)
                summaries.append(
                    dict(
                        graph=graph,
                        loss=loss,
                        metric=key,
                        pixel_columns=len(points),
                        rmse_eq26=float(np.sqrt(np.mean((p[:, 1] - p[:, 0]) ** 2))),
                        rmse_ratio=float(np.sqrt(np.mean((p[:, 2] - p[:, 0]) ** 2))),
                        max_abs_eq26=float(np.max(abs(p[:, 1] - p[:, 0]))),
                        max_abs_ratio=float(np.max(abs(p[:, 2] - p[:, 0]))),
                        half_linewidth_uncertainty=2 / (bottom - top) * (high - low),
                    )
                )
    (out / "raster_comparison.json").write_text(
        json.dumps(
            dict(note=raster_comparison.__doc__, summary=summaries, measurements=measurements),
            indent=2,
        )
    )
    return summaries


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=base / "data/reference/theory")
    parser.add_argument("--out", type=Path, default=base / "outputs/figures")
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional original Figure 2 raster (1806 x 896); used only for overlays.",
    )
    args = parser.parse_args()
    args.out = output_directory(args.out, parser)
    data = json.loads((args.results / "theory.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for convention in ["eq26", "ratio_of_means"]:
        figure(data, args.out, convention)
    if args.source:
        if Image.open(args.source).size != (1806, 896):
            parser.error("The source raster must have the original 1806 x 896 dimensions.")
        overlay(data, args.out, args.source)
        print(json.dumps(raster_comparison(data, args.out, args.source), indent=2))


if __name__ == "__main__":
    main()
