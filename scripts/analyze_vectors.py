
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from emotion_vectors import analysis
from emotion_vectors.config import TARGET_EMOTIONS
from emotion_vectors.vectors import build_emotion_vectors

DEFAULT_RUN_DIR = Path("data/runs/qwen35_2b_fp32")
DEFAULT_SEED = 42
FROZEN_LAYER = 12


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _activation_path(run_dir: Path, sample_id: str) -> Path:
    return run_dir / "activations" / f"{sample_id.replace('/', '_')}.pt"


def _load_from_run(run_dir: Path) -> dict[str, Any]:
    generations_path = run_dir / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(f"generation file does not exist: {generations_path}")

    all_records = _read_jsonl(generations_path)
    records = [
        record
        for record in all_records
        if record.get("validation", {}).get("valid", True)
    ]
    if not records:
        raise RuntimeError("no valid generations found")

    activations: dict[str, torch.Tensor] = {}
    for record in records:
        path = _activation_path(run_dir, record["sample_id"])
        if not path.exists():
            raise FileNotFoundError(
                f"missing activation for {record['sample_id']}: {path}"
            )
        activations[record["sample_id"]] = torch.load(
            path, map_location="cpu", weights_only=True
        ).float()

    raw, _unit, pair_ids = build_emotion_vectors(activations, records)
    return {
        "raw": raw,
        "records": records,
        "activations": activations,
        "pair_counts": {emotion: len(ids) for emotion, ids in pair_ids.items()},
        "record_count": len(records),
        "skipped_invalid": len(all_records) - len(records),
        "model_id": records[0].get("model_id"),
        "source": str(generations_path),
    }


def _load_from_vectors(vector_dir: Path) -> dict[str, Any]:
    tensor_path = vector_dir / "emotion_vectors.safetensors"
    if not tensor_path.exists():
        raise FileNotFoundError(f"vector artifact does not exist: {tensor_path}")

    metadata_path = vector_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    emotions = metadata.get("emotions") or list(TARGET_EMOTIONS)
    tensors = load_file(tensor_path, device="cpu")
    raw = {
        emotion: tensors[f"raw/{emotion}/all_layers"].float() for emotion in emotions
    }
    return {
        "raw": raw,
        "records": None,
        "activations": None,
        "pair_counts": metadata.get("pair_counts"),
        "record_count": metadata.get("valid_generation_count"),
        "skipped_invalid": None,
        "model_id": metadata.get("model_id"),
        "source": str(tensor_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def _print_heldout(result: dict[str, Any]) -> None:
    chance = result["chance_accuracy"]
    print(
        f"held-out split: {result['train_pair_count']} train groups, "
        f"{result['test_pair_count']} test groups, "
        f"{result['test_story_count']} test stories, chance {chance * 100:.0f}%"
    )
    header = f"{'layer':>5} {'accuracy':>9} {'random dirs':>12}"
    print(header)
    print("-" * len(header))
    for row in result["per_layer"]:
        marker = " *" if row["layer"] == result["best_layer"] else ""
        print(
            f"{row['layer']:>5} "
            f"{row['accuracy'] * 100:>8.1f}% "
            f"{row['random_direction_accuracy'] * 100:>11.1f}%"
            f"{marker}"
        )
    print("\n* best held-out layer")


def _plot_heldout(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["per_layer"]
    layers = [row["layer"] for row in rows]
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(
        layers,
        [row["accuracy"] * 100 for row in rows],
        marker="o",
        label="emotion vectors",
    )
    axes.plot(
        layers,
        [row["random_direction_accuracy"] * 100 for row in rows],
        linestyle=":",
        color="gray",
        label="random directions",
    )
    axes.axhline(
        result["chance_accuracy"] * 100,
        color="black",
        linewidth=1,
        linestyle="--",
        label=f"chance ({result['chance_accuracy'] * 100:.0f}%)",
    )
    axes.axvline(result["best_layer"], color="green", linewidth=1, alpha=0.5)
    axes.set_ylim(0, 100)
    axes.set_xlabel("decoder layer index (0 = output of the first decoder layer)")
    axes.set_ylabel("held-out accuracy (%)")
    axes.set_title(
        f"Held-out emotion classification, {result['test_story_count']} unseen stories"
    )
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_confusion(
    matrix: list[list[int]],
    emotions: list[str],
    layer: int,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    totals = [max(sum(row), 1) for row in matrix]
    shares = [
        [value / total for value in row] for row, total in zip(matrix, totals, strict=True)
    ]
    figure, axes = plt.subplots(figsize=(8, 7))
    image = axes.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axes)
    axes.set_xticks(range(len(emotions)), emotions, rotation=90)
    axes.set_yticks(range(len(emotions)), emotions)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            axes.text(
                j,
                i,
                str(matrix[i][j]),
                ha="center",
                va="center",
                fontsize=8,
                color="black" if shares[i][j] < 0.6 else "white",
            )
    axes.set_title(f"Held-out confusion matrix, layer {layer}")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_cosine_heatmap(
    cosine: torch.Tensor,
    emotions: list[str],
    layer: int,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = cosine.tolist()
    figure, axes = plt.subplots(figsize=(8, 7))
    image = axes.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    figure.colorbar(image, ax=axes)
    axes.set_xticks(range(len(emotions)), emotions, rotation=90)
    axes.set_yticks(range(len(emotions)), emotions)
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            axes.text(
                j,
                i,
                f"{values[i][j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    axes.set_title(f"Emotion vector cosine similarity, layer {layer}")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_pca(
    coordinates: torch.Tensor,
    ratios: list[float],
    emotions: list[str],
    layer: int,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = coordinates.tolist()
    figure, axes = plt.subplots(figsize=(7, 6))
    for emotion, point in zip(emotions, points, strict=True):
        color = "tab:red" if emotion in analysis.NEGATIVE_EMOTIONS else "tab:blue"
        axes.scatter(point[0], point[1], color=color, s=40)
        axes.annotate(
            emotion,
            (point[0], point[1]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=9,
        )
    axes.axhline(0, color="gray", linewidth=0.6)
    axes.axvline(0, color="gray", linewidth=0.6)
    axes.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}% variance)")
    axes.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}% variance)")
    axes.set_title(
        f"PCA of the {len(emotions)} emotion vectors, layer {layer}\n"
        "red = negative valence, blue = positive valence"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir is not None:
        loaded = _load_from_run(args.run_dir)
        default_output = args.run_dir / "analysis"
    else:
        loaded = _load_from_vectors(args.vector_dir)
        default_output = args.vector_dir / "analysis"

    raw = loaded["raw"]
    emotions = [emotion for emotion in TARGET_EMOTIONS if emotion in raw]
    missing = set(TARGET_EMOTIONS) - set(emotions)
    if missing:
        raise RuntimeError(f"missing vectors for emotions: {sorted(missing)}")

    vectors = {emotion: value.float() for emotion, value in raw.items()}
    layer_count, hidden_size = next(iter(vectors.values())).shape
    print(
        f"loaded {len(emotions)} emotion vectors, {layer_count} layers, "
        f"hidden size {hidden_size}"
    )
    if loaded["record_count"] is not None:
        print(f"valid generations used: {loaded['record_count']}")

    output_dir = args.output_dir or default_output
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------- held-out classification
    heldout: dict[str, Any] | None = None
    if loaded["activations"] is not None:
        print("\n=== held-out classification ===")
        heldout = analysis.held_out_classification(
            loaded["activations"],
            loaded["records"],
            emotions,
            seed=args.seed,
            train_fraction=args.train_fraction,
            reference=args.reference,
        )
        _print_heldout(heldout)
        _write_csv(
            output_dir / "heldout_accuracy.csv",
            heldout["per_layer"],
            ["layer", "accuracy", "random_direction_accuracy"],
        )
    else:
        print(
            "\nno activations available (--vector-dir mode), "
            "skipping held-out classification"
        )

    # ------------------------------------------------------ layers to inspect
    layers = list(args.layers) if args.layers else []
    if not layers:
        layers = [FROZEN_LAYER] if FROZEN_LAYER < layer_count else [layer_count // 2]
        if heldout is not None and heldout["best_layer"] not in layers:
            layers.append(heldout["best_layer"])
    for layer in layers:
        if not 0 <= layer < layer_count:
            raise ValueError(f"layer must be in [0, {layer_count - 1}], received {layer}")
    layers = sorted(set(layers))

    per_layer: dict[str, Any] = {}
    for layer in layers:
        matrix = analysis.stack_layer(vectors, layer, emotions)
        cosine = analysis.cosine_matrix(matrix)
        cosine_rows = [
            {"emotion": emotion, **dict(zip(emotions, values, strict=True))}
            for emotion, values in zip(emotions, cosine.tolist(), strict=True)
        ]
        _write_csv(
            output_dir / f"cosine_matrix_layer{layer}.csv",
            cosine_rows,
            ["emotion", *emotions],
        )

        result = analysis.pca(analysis.unit_rows(matrix))
        ratios = result["explained_variance_ratio"]
        coordinates = result["coordinates"]
        component_names = [f"pc{index + 1}" for index in range(coordinates.shape[1])]
        pca_rows = [
            {"emotion": emotion, **dict(zip(component_names, point, strict=True))}
            for emotion, point in zip(emotions, coordinates.tolist(), strict=True)
        ]
        _write_csv(
            output_dir / f"pca_layer{layer}.csv",
            pca_rows,
            ["emotion", *component_names],
        )

        pc1_valence = analysis.axis_valence_separation(coordinates[:, 0], emotions)
        entry: dict[str, Any] = {
            "cosine": analysis.cosine_summary(cosine, emotions),
            "pca": {
                "explained_variance_ratio": ratios,
                "cumulative_first_two": sum(ratios[:2]),
                "pc1_valence_separation": pc1_valence,
            },
        }

        print(f"\n--- layer {layer} ---")
        summary = entry["cosine"]
        print(
            f"mean pairwise cosine {_format(summary['mean_pairwise_cosine'])}, "
            f"negative pairs {summary['negative_cosine_pair_count']}/{summary['pair_count']}"
        )
        print(
            f"within negative {_format(summary['within_negative_cosine'])}, "
            f"within positive {_format(summary['within_positive_cosine'])}, "
            f"across valence {_format(summary['across_valence_cosine'])}"
        )
        print(
            f"most similar pair {summary['most_similar_pair']}, "
            f"least similar pair {summary['least_similar_pair']}"
        )
        print(
            "PCA explained variance: "
            + ", ".join(
                f"PC{index + 1} {ratio * 100:.1f}%"
                for index, ratio in enumerate(ratios[:4])
            )
        )
        print(
            f"PC1 valence sign accuracy {pc1_valence['sign_accuracy'] * 100:.0f}% "
            f"({len(emotions)} emotions)"
        )

        if heldout is not None:
            predicted = heldout["_predictions"][layer]
            labels = heldout["_labels"]
            confusion = analysis.confusion_matrix(predicted, labels, emotions)
            recall = analysis.per_emotion_accuracy(predicted, labels, emotions)
            entry["heldout"] = {
                "accuracy": heldout["accuracy_by_layer"][layer],
                "per_emotion": recall,
                "confusion_matrix": confusion,
                "confusion_row_labels": emotions,
                "confusion_column_labels": emotions,
            }
            confusion_rows = [
                {"true": emotion, **dict(zip(emotions, row, strict=True))}
                for emotion, row in zip(emotions, confusion, strict=True)
            ]
            _write_csv(
                output_dir / f"heldout_confusion_layer{layer}.csv",
                confusion_rows,
                ["true", *emotions],
            )
            print(
                f"held-out accuracy {heldout['accuracy_by_layer'][layer] * 100:.1f}% "
                f"on {heldout['test_story_count']} unseen stories"
            )

        per_layer[str(layer)] = entry

        if not args.no_figures:
            try:
                _plot_cosine_heatmap(
                    cosine,
                    emotions,
                    layer,
                    figure_dir / f"cosine_heatmap_layer{layer}.png",
                )
                _plot_pca(
                    coordinates,
                    ratios,
                    emotions,
                    layer,
                    figure_dir / f"pca_layer{layer}.png",
                )
                if heldout is not None:
                    _plot_confusion(
                        entry["heldout"]["confusion_matrix"],
                        emotions,
                        layer,
                        figure_dir / f"heldout_confusion_layer{layer}.png",
                    )
            except ImportError:
                print("matplotlib is not installed, skipping figures")
                args.no_figures = True

    if heldout is not None and not args.no_figures:
        try:
            _plot_heldout(heldout, figure_dir / "heldout_accuracy.png")
        except ImportError:
            print("matplotlib is not installed, skipping figures")

    heldout_summary = None
    if heldout is not None:
        heldout_summary = {
            key: value
            for key, value in heldout.items()
            if not key.startswith("_")
        }

    summary = {
        "format_version": 2,
        "vector_definition": (
            "mean paired difference: emotion story activation minus the "
            "matched neutral story activation of the same pair_id"
        ),
        "source": loaded["source"],
        "model_id": loaded["model_id"],
        "emotions": emotions,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "valid_generation_count": loaded["record_count"],
        "skipped_invalid_count": loaded["skipped_invalid"],
        "pair_counts": loaded["pair_counts"],
        "seed": args.seed,
        "frozen_layer": FROZEN_LAYER,
        "layers_analyzed": layers,
        "held_out_classification": heldout_summary,
        "random_direction_reference": analysis.random_direction_cosine(
            len(emotions), hidden_size, seed=args.seed
        ),
        "per_layer": per_layer,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nsaved analysis to {output_dir}")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out classification, cosine structure and PCA for emotion vectors."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory with generations.jsonl and activations/. "
            f"Defaults to {DEFAULT_RUN_DIR} when --vector-dir is not given."
        ),
    )
    source.add_argument(
        "--vector-dir",
        type=Path,
        default=None,
        help=(
            "Directory with emotion_vectors.safetensors from build_vectors.py. "
            "Held-out classification is unavailable in this mode."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layers for the heatmap, PCA and confusion matrix. Default: 12 plus the best held-out layer.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Share of pair_ids used to build the vectors. The rest is the held-out test set.",
    )
    parser.add_argument(
        "--reference",
        choices=("paired", "train_mean"),
        default="paired",
        help=(
            "What to subtract from a held-out story: its matched neutral story "
            "(paired, default) or the mean training activation (train_mean)."
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.run_dir is None and args.vector_dir is None:
        args.run_dir = DEFAULT_RUN_DIR
    run(args)


if __name__ == "__main__":
    main()
