"""Figures for the Dataset B Emotion Theater experiment.

Reads the artifacts written by `scripts/analyze_theater.py` and renders five
presentation figures. Recomputes nothing, so the figures and the reported
numbers cannot drift apart.

Usage:
    uv run python scripts/plot_theater.py
    uv run python scripts/plot_theater.py --run-dir data/runs/dataset_b --dpi 200
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUN_DIR = Path("data/runs/dataset_b")
INK = "#1f2933"
MUTED = "#7b8794"
RAW = "#cbd2d9"
CORRECTED = "#2f6f8f"
WARM = "#b4532a"


def _style(ax, *, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis=grid_axis, color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_offset(summary: dict, out: Path, dpi: int) -> Path:
    """Why the probe looked broken: constants larger than the signal."""
    offsets = summary["offset"]["internal_offsets"]
    scales = summary["offset"]["internal_scales"]
    order = sorted(offsets, key=lambda e: -offsets[e])
    x = range(len(order))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x, [offsets[e] for e in order], color=WARM, label="per-emotion constant")
    ax.bar(
        x,
        [scales[e] for e in order],
        color=CORRECTED,
        width=0.45,
        label="within-emotion sd (the real signal)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("projection units")
    ax.set_title(
        "A story-derived neutral baseline leaves a constant in every projection\n"
        f"spread {summary['offset']['offset_spread']:.3f} vs sd "
        f"{summary['offset']['mean_within_emotion_sd']:.3f} "
        f"({summary['offset']['ratio']:.1f}x)",
        fontsize=11,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    path = out / "fig1_domain_offset.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def fig_accuracy(summary: dict, out: Path, dpi: int) -> Path:
    """Removing the constant takes the probe from 26.6% to 66.1%."""
    acc = summary["accuracy"]
    chance = acc["chance"]
    labels, raw, corrected = [], [], []
    # Arm names vary by experiment: natural/suppression for the from-start
    # design, natural_placebo/switch for the mid-conversation one. The summary
    # records which pair was compared.
    for condition in (summary["control_arm"], summary["treatment_arm"]):
        for channel in ("internal", "external"):
            labels.append(f"{channel}\n{condition}")
            raw.append(acc[condition][channel]["raw"]["exact"] * 100)
            corrected.append(acc[condition][channel]["corrected"]["exact"] * 100)

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar([i - width / 2 for i in x], raw, width, color=RAW, label="raw projection")
    ax.bar(
        [i + width / 2 for i in x],
        corrected,
        width,
        color=CORRECTED,
        label="offset corrected",
    )
    ax.axhline(chance * 100, color=WARM, linestyle="--", linewidth=1.2)
    ax.text(
        len(labels) - 0.45,
        chance * 100 + 1.5,
        f"chance {chance:.0%}",
        color=WARM,
        fontsize=9,
        ha="right",
    )
    for i, (r, c) in enumerate(zip(raw, corrected)):
        ax.text(i - width / 2, r + 1.2, f"{r:.0f}", ha="center", fontsize=8, color=MUTED)
        ax.text(i + width / 2, c + 1.2, f"{c:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("emotion recovered, % of replies")
    ax.set_ylim(0, 80)
    ax.set_title(
        "Recovery accuracy, argmax over 10 emotions\n"
        "after correction the activation probe beats the blinded text judge",
        fontsize=11,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    path = out / "fig2_recovery_accuracy.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def fig_confusion(run_dir: Path, out: Path, dpi: int) -> Path:
    """Before and after, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, variant, title in zip(
        axes,
        ("raw", "corrected"),
        ("Raw projection: three attractors", "Offset corrected: a working probe"),
    ):
        path = run_dir / "analysis" / f"theater_confusion_{variant}.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        header, body = rows[0][1:], rows[1:]
        emotions = [r[0] for r in body]
        matrix = []
        for r in body:
            counts = [float(v) for v in r[1:]]
            total = sum(counts) or 1.0
            matrix.append([c / total * 100 for c in counts])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(header)))
        ax.set_xticklabels(header, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(emotions)))
        ax.set_yticklabels(emotions, fontsize=8)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("predicted", fontsize=9)
        for i in range(len(emotions)):
            for j in range(len(header)):
                v = matrix[i][j]
                if v >= 10:
                    ax.text(
                        j,
                        i,
                        f"{v:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if v > 55 else INK,
                    )
    axes[0].set_ylabel("true emotion", fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="% of replies")
    path = out / "fig3_confusion.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_result(summary: dict, out: Path, dpi: int) -> Path:
    """The headline: both channels drop, the internal one drops more."""
    tests = summary["result"]["tests"]
    names = ["R_internal", "R_external"]
    values = [tests[n]["mean"] for n in names]
    errs = [
        [tests[n]["mean"] - tests[n]["ci95"][0] for n in names],
        [tests[n]["ci95"][1] - tests[n]["mean"] for n in names],
    ]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.3), width_ratios=[1, 1.35])

    ax.bar(
        [0, 1],
        values,
        color=[CORRECTED, WARM],
        yerr=errs,
        capsize=5,
        ecolor=MUTED,
        width=0.55,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["internal\n(activations)", "external\n(blinded judge)"], fontsize=9)
    ax.set_ylabel("drop under suppression, z units")
    d = tests["D"]
    ax.set_title(
        f"D = {d['mean']:+.3f}  [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]\n"
        f"p = {d['p_value']:.0e}, negative in {d['n_negative']}/{d['n']} conversations",
        fontsize=10,
        color=INK,
    )
    _style(ax)

    per_emotion = summary["breakdowns"]["per_emotion"]
    order = sorted(per_emotion, key=lambda e: per_emotion[e]["mean_D"])
    bx.barh(
        range(len(order)),
        [per_emotion[e]["mean_D"] for e in order],
        color=[WARM if per_emotion[e]["mean_D"] < -0.5 else CORRECTED for e in order],
        height=0.62,
    )
    bx.axvline(0, color=MUTED, linewidth=1)
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels(order, fontsize=8)
    bx.set_xlabel("mean D per emotion (20 conversations each)")
    bx.set_title(
        "Concentrated in negative, low-arousal emotions; absent for joy",
        fontsize=10,
        color=INK,
    )
    _style(bx, grid_axis="x")

    fig.tight_layout()
    path = out / "fig4_result.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def fig_turn(summary: dict, out: Path, dpi: int) -> Path:
    """The two channels converge as the scenario escalates."""
    per_turn = summary["breakdowns"]["per_turn"]
    turns = sorted(per_turn, key=lambda t: int(t))
    values = [per_turn[t]["mean_D"] for t in turns]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(
        [int(t) for t in turns],
        values,
        marker="o",
        color=CORRECTED,
        linewidth=2,
        markersize=7,
    )
    for t, v in zip(turns, values):
        ax.annotate(
            f"{v:+.2f}",
            (int(t), v),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xticks([int(t) for t in turns])
    ax.set_xlabel("conversation turn (situation escalates left to right)")
    ax.set_ylabel("mean D, z units")
    ax.set_title(
        "The gap narrows as pressure builds\n"
        "internal and expressed emotion converge over the five turns",
        fontsize=11,
        color=INK,
    )
    _style(ax)
    fig.tight_layout()
    path = out / "fig5_by_turn.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    summary_path = run_dir / "analysis" / "theater_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found. Run scripts/analyze_theater.py first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    out = run_dir / "analysis" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    for builder in (fig_offset, fig_accuracy, fig_result, fig_turn):
        print(f"wrote {builder(summary, out, args.dpi)}")
    print(f"wrote {fig_confusion(run_dir, out, args.dpi)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Dataset B figures.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
