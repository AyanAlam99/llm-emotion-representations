
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

CLASSIFIER_ID = "j-hartmann/emotion-english-distilroberta-base"
SIMILARITY_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Our GoEmotions-style labels mapped onto the classifier's 7 labels.
LABEL_MAP = {
    "joy": "joy",
    "excitement": "joy",
    "admiration": "joy",
    "gratitude": "joy",
    "sadness": "sadness",
    "disappointment": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "fear": "fear",
    "nervousness": "fear",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _classify(texts: list[str], batch_size: int) -> list[dict[str, float]]:
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model=CLASSIFIER_ID,
        top_k=None,
        truncation=True,
        device=-1,
    )
    scored: list[dict[str, float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        for result in classifier(chunk):
            scored.append({item["label"]: float(item["score"]) for item in result})
    return scored


def _similarities(
    texts: list[str],
    references: list[str],
) -> list[float] | None:
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print(
            "sentence-transformers is not installed, skipping semantic similarity "
            "(pip install sentence-transformers)"
        )
        return None

    encoder = SentenceTransformer(SIMILARITY_ID)
    left = encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    right = encoder.encode(references, convert_to_tensor=True, normalize_embeddings=True)
    return [float(util.cos_sim(a, b)) for a, b in zip(left, right, strict=True)]


def _group_key(row: dict[str, Any]) -> tuple[str, int, str, float]:
    # A baseline row has no layer, so it sorts to -1 and keeps its own group.
    layer = row.get("layer")
    return (
        row["condition"],
        -1 if layer is None else int(layer),
        row["emotion"] or "-",
        float(row["alpha"]),
    )


def summarise(
    rows: list[dict[str, Any]],
    prompt_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """One row per (condition, layer, emotion, alpha), averaged over the prompts.

    `prompt_indices` restricts the average to a subset of prompts, which is how
    the layer-selection split keeps its reporting half untouched.
    """
    if prompt_indices is not None:
        wanted = set(prompt_indices)
        rows = [row for row in rows if row["prompt_index"] in wanted]

    groups: dict[tuple[str, int, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    summary: list[dict[str, Any]] = []
    for (condition, layer, emotion, alpha), items in sorted(groups.items()):
        deltas = [
            item["target_probability_delta"]
            for item in items
            if item.get("target_probability_delta") is not None
        ]
        similarities = [
            item["baseline_similarity"]
            for item in items
            if item.get("baseline_similarity") is not None
        ]
        summary.append(
            {
                "condition": condition,
                "layer": None if layer < 0 else layer,
                "emotion": emotion,
                "alpha": alpha,
                "prompt_count": len(items),
                "target_probability": round(
                    mean(item["target_probability"] for item in items), 4
                ),
                "target_probability_delta": round(mean(deltas), 4) if deltas else None,
                "increase_rate": round(
                    sum(1 for value in deltas if value > 0) / len(deltas), 4
                )
                if deltas
                else None,
                "top_label_is_target_rate": round(
                    mean(1.0 if item["top_label_is_target"] else 0.0 for item in items),
                    4,
                ),
                "baseline_similarity": round(mean(similarities), 4)
                if similarities
                else None,
                "repeated_trigram_rate": round(
                    mean(item["metrics"]["repeated_trigram_rate"] for item in items), 4
                ),
                "cjk_share": round(
                    mean(item["metrics"]["cjk_share"] for item in items), 4
                ),
                "word_count": round(
                    mean(item["metrics"]["word_count"] for item in items), 1
                ),
            }
        )
    return summary


def layer_scores(
    summary: list[dict[str, Any]],
    *,
    alphas: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate a summary down to one row per layer.

    `emotion_gain` averages the target-probability delta over every emotion and
    alpha. `control_gain` is the same for the random-vector rows, so the
    difference is the part attributable to the emotion direction rather than to
    perturbing the residual stream at all.
    """
    layers = sorted({row["layer"] for row in summary if row["layer"] is not None})
    out: list[dict[str, Any]] = []
    for layer in layers:
        rows = [row for row in summary if row["layer"] == layer]
        if alphas is not None:
            keep = set(alphas)
            rows = [row for row in rows if row["alpha"] in keep]
        emotion_rows = [row for row in rows if row["condition"] == "emotion"]
        control_rows = [row for row in rows if row["condition"] == "random"]
        if not emotion_rows:
            continue
        gain = mean(row["target_probability_delta"] or 0.0 for row in emotion_rows)
        control = (
            mean(row["target_probability_delta"] or 0.0 for row in control_rows)
            if control_rows
            else 0.0
        )
        sims = [
            row["baseline_similarity"]
            for row in emotion_rows
            if row["baseline_similarity"] is not None
        ]
        out.append(
            {
                "layer": layer,
                "emotion_gain": round(gain, 4),
                "control_gain": round(control, 4),
                "net_gain": round(gain - control, 4),
                "top_label_is_target_rate": round(
                    mean(row["top_label_is_target_rate"] for row in emotion_rows), 4
                ),
                "baseline_similarity": round(mean(sims), 4) if sims else None,
                "repeated_trigram_rate": round(
                    mean(row["repeated_trigram_rate"] for row in emotion_rows), 4
                ),
                "cjk_share": round(
                    mean(row["cjk_share"] for row in emotion_rows), 4
                ),
            }
        )
    return out


def select_layer(
    rows: list[dict[str, Any]],
    *,
    select_prompts: Sequence[int],
    alphas: Sequence[float] | None,
    min_similarity: float,
    max_repeated_trigram: float,
) -> dict[str, Any]:
    """Pick the steering layer by a rule fixed before the run.

    Rule: among layers whose steered text stays coherent (mean similarity to the
    unsteered response at least `min_similarity`, mean repeated-trigram rate at
    most `max_repeated_trigram`), take the largest net emotion gain. If no layer
    clears the coherence bar, fall back to the best net gain and say so.

    Only `select_prompts` are used, so the remaining prompts stay untouched for
    reporting.
    """
    chosen_summary = summarise(rows, prompt_indices=select_prompts)
    scores = layer_scores(chosen_summary, alphas=alphas)
    eligible = [
        row
        for row in scores
        if (row["baseline_similarity"] is None
            or row["baseline_similarity"] >= min_similarity)
        and row["repeated_trigram_rate"] <= max_repeated_trigram
    ]
    pool = eligible or scores
    best = max(pool, key=lambda row: row["net_gain"]) if pool else None
    return {
        "rule": (
            f"max net_gain subject to baseline_similarity >= {min_similarity} "
            f"and repeated_trigram_rate <= {max_repeated_trigram}"
        ),
        "select_prompts": list(select_prompts),
        "alphas_considered": list(alphas) if alphas else "all",
        "candidates": scores,
        "eligible_layers": [row["layer"] for row in eligible],
        "coherence_bar_met": bool(eligible),
        "selected_layer": best["layer"] if best else None,
    }


def _plot_dose_response(
    summary: list[dict[str, Any]],
    emotions: list[str],
    path: Path,
) -> None:
    """One dose-response panel per layer, emotion deltas plus the random control."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({row["layer"] for row in summary if row["layer"] is not None})
    if not layers:
        return
    columns = min(len(layers), 4)
    rows_n = (len(layers) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows_n, columns, figsize=(5 * columns, 4.4 * rows_n), squeeze=False
    )
    limits = []
    for position, layer in enumerate(layers):
        ax = axes[position // columns][position % columns]
        for emotion in emotions:
            points = sorted(
                (
                    row
                    for row in summary
                    if row["layer"] == layer and row["emotion"] == emotion
                ),
                key=lambda row: row["alpha"],
            )
            if not points:
                continue
            values = [0.0] + [row["target_probability_delta"] or 0.0 for row in points]
            limits.extend(values)
            ax.plot(
                [0.0] + [row["alpha"] for row in points],
                values,
                marker="o",
                label=emotion,
            )
        control = sorted(
            (
                row
                for row in summary
                if row["layer"] == layer and row["condition"] == "random"
            ),
            key=lambda row: row["alpha"],
        )
        if control:
            ax.plot(
                [0.0] + [row["alpha"] for row in control],
                [0.0] + [row["target_probability_delta"] or 0.0 for row in control],
                marker="x",
                linestyle=":",
                color="gray",
                label="random vector (same norm)",
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"layer {layer}")
        ax.set_xlabel("alpha")
        if position % columns == 0:
            ax.set_ylabel("target emotion probability minus baseline")
        ax.legend(fontsize=8)
    if limits:
        low, high = min(limits), max(limits)
        pad = 0.05 * max(high - low, 0.1)
        for position in range(len(layers)):
            axes[position // columns][position % columns].set_ylim(
                low - pad, high + pad
            )
    for position in range(len(layers), rows_n * columns):
        axes[position // columns][position % columns].axis("off")
    figure.suptitle("Does steering produce the target emotion, by layer?")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_layer_comparison(scores: list[dict[str, Any]], path: Path) -> None:
    """Net emotion gain and text coherence against layer, side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not scores:
        return
    layers = [row["layer"] for row in scores]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    left.plot(layers, [row["emotion_gain"] for row in scores], marker="o",
              label="emotion vectors")
    left.plot(layers, [row["control_gain"] for row in scores], marker="x",
              linestyle=":", color="gray", label="random vector (same norm)")
    left.axhline(0, color="black", linewidth=0.8)
    left.set_xticks(layers)
    left.set_xlabel("steered decoder layer")
    left.set_ylabel("target emotion probability minus baseline")
    left.set_title("Steering effect by layer")
    left.legend()

    sims = [row["baseline_similarity"] for row in scores]
    if all(value is not None for value in sims):
        right.plot(layers, sims, marker="o", label="similarity to unsteered")
    right.plot(layers, [row["repeated_trigram_rate"] for row in scores], marker="s",
               label="repeated 3-gram rate")
    right.plot(layers, [row["cjk_share"] for row in scores], marker="^",
               label="CJK character share")
    right.set_ylim(0, 1.05)
    right.set_xticks(layers)
    right.set_xlabel("steered decoder layer")
    right.set_title("Text quality by layer")
    right.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _print_ladder(
    scored: list[dict[str, Any]],
    prompt_index: int,
    emotion: str,
    layer: int | None = None,
) -> None:
    """Print one prompt's responses from alpha 0 upward, for a slide or a demo.

    This is the single most convincing thing to show a human: identical prompt,
    identical greedy decoding, only the injected vector changes.
    """
    ladder = [
        row
        for row in scored
        if row["prompt_index"] == prompt_index
        and (
            row["condition"] == "baseline"
            or (row["emotion"] == emotion and (layer is None or row["layer"] == layer))
        )
    ]
    if not ladder:
        print(f"\nno rows for prompt {prompt_index} / emotion {emotion}")
        return
    ladder.sort(key=lambda row: row["alpha"])
    # The baseline row carries no emotion of its own, so read every row against
    # the ladder's label to keep the column comparable from top to bottom.
    label = LABEL_MAP[emotion]
    where = "" if layer is None else f", layer {layer}"
    print(f"\n=== alpha ladder: {emotion}{where}, prompt {prompt_index} ===")
    print(f"prompt: {ladder[0]['prompt']}\n")
    for row in ladder:
        probability = row["classifier_distribution"].get(label, 0.0)
        print(
            f"--- alpha {row['alpha']:g}  "
            f"p({label}) = {probability:.3f}  "
            f"repeated-3gram {row['metrics']['repeated_trigram_rate']:.2f} ---"
        )
        print(row["response"])
        print()


def _print_layer_comparison(
    scored: list[dict[str, Any]],
    prompt_index: int,
    emotion: str,
    alpha: float,
) -> None:
    """Same prompt, same emotion, same strength, one response per layer.

    This is the qualitative counterpart to the by-layer figure: it shows what
    "steering works better here" actually looks like as text.
    """
    label = LABEL_MAP[emotion]
    baseline = [
        row
        for row in scored
        if row["prompt_index"] == prompt_index and row["condition"] == "baseline"
    ]
    rows = sorted(
        (
            row
            for row in scored
            if row["prompt_index"] == prompt_index
            and row["condition"] == "emotion"
            and row["emotion"] == emotion
            and abs(float(row["alpha"]) - alpha) < 1e-9
        ),
        key=lambda row: row["layer"],
    )
    if not rows:
        print(f"\nno rows for prompt {prompt_index}, {emotion}, alpha {alpha:g}")
        return
    print(f"\n=== {emotion} at alpha {alpha:g}, across layers, prompt {prompt_index} ===")
    print(f"prompt: {rows[0]['prompt']}\n")
    for row in baseline:
        print(f"--- unsteered  p({label}) = "
              f"{row['classifier_distribution'].get(label, 0.0):.3f} ---")
        print(row["response"])
        print()
    for row in rows:
        print(
            f"--- layer {row['layer']}  "
            f"p({label}) = {row['classifier_distribution'].get(label, 0.0):.3f}  "
            f"sim {row['baseline_similarity'] if row['baseline_similarity'] is not None else float('nan'):.2f}  "
            f"repeated-3gram {row['metrics']['repeated_trigram_rate']:.2f} ---"
        )
        print(row["response"])
        print()


def run(args: argparse.Namespace) -> None:
    generations_path = args.steering_dir / "generations.jsonl"
    rows = _read_jsonl(generations_path)
    if not rows:
        raise RuntimeError(f"no rows in {generations_path}")
    print(f"scoring {len(rows)} responses from {generations_path}")

    texts = [row["response"] for row in rows]
    distributions = _classify(texts, args.batch_size)

    baseline_by_prompt = {
        row["prompt_index"]: distribution
        for row, distribution in zip(rows, distributions, strict=True)
        if row["condition"] == "baseline"
    }
    baseline_text_by_prompt = {
        row["prompt_index"]: row["response"]
        for row in rows
        if row["condition"] == "baseline"
    }

    similarities = None
    if not args.no_similarity:
        references = [baseline_text_by_prompt[row["prompt_index"]] for row in rows]
        similarities = _similarities(texts, references)

    scored: list[dict[str, Any]] = []
    for index, (row, distribution) in enumerate(
        zip(rows, distributions, strict=True)
    ):
        # A random-control row has no emotion of its own, so it is scored
        # against the same label as the emotion rows it is compared with.
        emotion = row["emotion"] or args.control_label
        label = LABEL_MAP.get(emotion)
        if label is None:
            raise KeyError(f"no classifier label mapped for emotion {emotion!r}")

        probability = distribution.get(label, 0.0)
        baseline = baseline_by_prompt.get(row["prompt_index"])
        top_label = max(distribution.items(), key=lambda item: item[1])[0]
        scored.append(
            {
                **row,
                "classifier_label": label,
                "classifier_distribution": {
                    key: round(value, 4) for key, value in distribution.items()
                },
                "target_probability": round(probability, 4),
                "target_probability_delta": round(
                    probability - baseline.get(label, 0.0), 4
                )
                if baseline is not None
                else None,
                "top_label": top_label,
                "top_label_is_target": top_label == label,
                "baseline_similarity": round(similarities[index], 4)
                if similarities is not None
                else None,
            }
        )

    _write_jsonl(args.steering_dir / "scores.jsonl", scored)

    summary = summarise(scored)
    emotions = sorted(
        {row["emotion"] for row in summary if row["condition"] == "emotion"}
    )
    all_prompts = sorted({row["prompt_index"] for row in scored})
    layers = sorted({row["layer"] for row in summary if row["layer"] is not None})

    # Layer selection on one half of the prompts, reporting on the other half.
    selection = None
    report_prompts = all_prompts
    if args.select_split and len(layers) > 1:
        split = min(args.select_split, len(all_prompts) - 1)
        select_prompts = all_prompts[:split]
        report_prompts = all_prompts[split:]
        selection = select_layer(
            scored,
            select_prompts=select_prompts,
            alphas=args.select_alphas,
            min_similarity=args.min_similarity,
            max_repeated_trigram=args.max_repeated_trigram,
        )
        selection["report_prompts"] = report_prompts

    report_summary = (
        summarise(scored, prompt_indices=report_prompts)
        if selection is not None
        else summary
    )
    scores_by_layer = layer_scores(summary, alphas=args.select_alphas)

    payload = {
        "format_version": 2,
        "source": str(generations_path),
        "classifier_id": CLASSIFIER_ID,
        "similarity_id": SIMILARITY_ID if similarities is not None else None,
        "label_map": LABEL_MAP,
        "control_label": args.control_label,
        "response_count": len(scored),
        "emotions": emotions,
        "layers": layers,
        "layer_scores": scores_by_layer,
        "layer_selection": selection,
        "groups": summary,
        "groups_report_prompts_only": report_summary if selection else None,
    }
    (args.steering_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    header = (
        f"{'layer':>6} {'condition':>10} {'emotion':>12} {'alpha':>6} {'p(target)':>10} "
        f"{'delta':>8} {'up rate':>8} {'top=tgt':>8} {'sim':>6} {'rep3':>6} {'cjk':>6}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in summary:
        layer_text = "base" if row["layer"] is None else str(row["layer"])
        print(
            f"{layer_text:>6} {row['condition']:>10} {row['emotion']:>12} "
            f"{row['alpha']:>6.2f} "
            f"{row['target_probability']:>10.3f} "
            f"{(row['target_probability_delta'] or 0.0):>+8.3f} "
            f"{(row['increase_rate'] or 0.0):>8.2f} "
            f"{row['top_label_is_target_rate']:>8.2f} "
            f"{(row['baseline_similarity'] or 0.0):>6.2f} "
            f"{row['repeated_trigram_rate']:>6.2f} "
            f"{row['cjk_share']:>6.2f}"
        )

    if scores_by_layer:
        head = (
            f"\n{'layer':>6} {'emotion':>9} {'control':>9} {'net':>8} "
            f"{'top=tgt':>8} {'sim':>6} {'rep3':>6} {'cjk':>6}"
        )
        print(head)
        print("-" * (len(head) - 1))
        for row in scores_by_layer:
            print(
                f"{row['layer']:>6} {row['emotion_gain']:>+9.3f} "
                f"{row['control_gain']:>+9.3f} {row['net_gain']:>+8.3f} "
                f"{row['top_label_is_target_rate']:>8.2f} "
                f"{(row['baseline_similarity'] or 0.0):>6.2f} "
                f"{row['repeated_trigram_rate']:>6.2f} {row['cjk_share']:>6.2f}"
            )

    if selection is not None:
        print(f"\n=== layer selection ===\nrule: {selection['rule']}")
        print(f"selected on prompts {selection['select_prompts']}, "
              f"reported on prompts {selection['report_prompts']}")
        if not selection["coherence_bar_met"]:
            print("WARNING: no layer met the coherence bar; fell back to best net gain")
        print(f"selected layer: {selection['selected_layer']}")
        chosen = [
            row
            for row in report_summary
            if row["layer"] == selection["selected_layer"]
            and row["condition"] == "emotion"
        ]
        if chosen:
            print(
                "held-out prompts at the selected layer: mean delta "
                f"{mean(row['target_probability_delta'] or 0.0 for row in chosen):+.3f}"
            )

    if args.show_prompt is not None:
        if args.show_alpha is not None:
            _print_layer_comparison(
                scored, args.show_prompt, args.show_emotion, args.show_alpha
            )
        else:
            _print_ladder(scored, args.show_prompt, args.show_emotion, args.show_layer)

    if not args.no_figures:
        figure_dir = args.steering_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_dose_response(
                summary, emotions, figure_dir / "steering_dose_response.png"
            )
            _plot_layer_comparison(
                scores_by_layer, figure_dir / "steering_by_layer.png"
            )
        except ImportError:
            print("matplotlib is not installed, skipping figures")

    print(f"\nsaved scores to {args.steering_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score steered responses.")
    parser.add_argument(
        "--steering-dir",
        type=Path,
        default=Path("data/runs/qwen35_2b_fp32/steering"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--control-label",
        default="joy",
        help=(
            "Which target label the random-vector control rows are scored "
            "against. Any choice is arbitrary; it only sets the null curve."
        ),
    )
    parser.add_argument("--no-similarity", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--select-split",
        type=int,
        default=0,
        help=(
            "Choose the steering layer using this many prompts and report on "
            "the rest. 0 disables selection and reports on everything."
        ),
    )
    parser.add_argument(
        "--select-alphas",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Restrict layer selection to these alphas, e.g. 0.5 1.0 to ignore "
            "strengths that are known to break the text. Default: all."
        ),
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.4,
        help="Coherence bar: mean similarity to the unsteered response.",
    )
    parser.add_argument(
        "--max-repeated-trigram",
        type=float,
        default=0.25,
        help="Coherence bar: mean repeated 3-gram rate.",
    )
    parser.add_argument(
        "--show-prompt",
        type=int,
        default=None,
        help="Print every alpha for this prompt index, baseline first. For slides.",
    )
    parser.add_argument(
        "--show-emotion",
        default="sadness",
        help="Which emotion's ladder --show-prompt prints.",
    )
    parser.add_argument(
        "--show-layer",
        type=int,
        default=None,
        help="Restrict the --show-prompt ladder to one layer.",
    )
    parser.add_argument(
        "--show-alpha",
        type=float,
        default=None,
        help=(
            "With --show-prompt, print one response per layer at this alpha "
            "instead of the alpha ladder. The qualitative layer comparison."
        ),
    )
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
