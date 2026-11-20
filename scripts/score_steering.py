
from __future__ import annotations

import argparse
import json
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


def _group_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (row["condition"], row["emotion"] or "-", float(row["alpha"]))


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (condition, emotion, alpha), averaged over the prompts."""
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    summary: list[dict[str, Any]] = []
    for (condition, emotion, alpha), items in sorted(groups.items()):
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


def _plot_dose_response(
    summary: list[dict[str, Any]],
    emotions: list[str],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    for emotion in emotions:
        points = sorted(
            (row for row in summary if row["emotion"] == emotion),
            key=lambda row: row["alpha"],
        )
        if not points:
            continue
        left.plot(
            [0.0] + [row["alpha"] for row in points],
            [0.0] + [row["target_probability_delta"] or 0.0 for row in points],
            marker="o",
            label=emotion,
        )
    control = sorted(
        (row for row in summary if row["condition"] == "random"),
        key=lambda row: row["alpha"],
    )
    if control:
        left.plot(
            [0.0] + [row["alpha"] for row in control],
            [0.0] + [row["target_probability_delta"] or 0.0 for row in control],
            marker="x",
            linestyle=":",
            color="gray",
            label="random vector (same norm)",
        )
    left.axhline(0, color="black", linewidth=0.8)
    left.set_xlabel("alpha (fraction of the layer's typical token norm)")
    left.set_ylabel("target emotion probability, minus unsteered baseline")
    left.set_title("Does steering produce the target emotion?")
    left.legend()

    plotted = 0
    for emotion in emotions:
        points = sorted(
            (row for row in summary if row["emotion"] == emotion),
            key=lambda row: row["alpha"],
        )
        if not points or points[0]["baseline_similarity"] is None:
            continue
        right.plot(
            [0.0] + [row["alpha"] for row in points],
            [1.0] + [row["baseline_similarity"] for row in points],
            marker="o",
            label=emotion,
        )
        plotted += 1
    right.set_ylim(0, 1.05)
    right.set_xlabel("alpha (fraction of the layer's typical token norm)")
    right.set_ylabel("cosine similarity to the unsteered response")
    if plotted:
        right.set_title("How much of the original response survives?")
        right.legend()
    else:
        right.set_title(
            "Semantic similarity unavailable\n(install sentence-transformers)"
        )

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _print_ladder(
    scored: list[dict[str, Any]],
    prompt_index: int,
    emotion: str,
) -> None:
    """Print one prompt's responses from alpha 0 upward, for a slide or a demo.

    This is the single most convincing thing to show a human: identical prompt,
    identical greedy decoding, only the injected vector changes.
    """
    ladder = [
        row
        for row in scored
        if row["prompt_index"] == prompt_index
        and (row["condition"] == "baseline" or row["emotion"] == emotion)
    ]
    if not ladder:
        print(f"\nno rows for prompt {prompt_index} / emotion {emotion}")
        return
    ladder.sort(key=lambda row: row["alpha"])
    # The baseline row carries no emotion of its own, so read every row against
    # the ladder's label to keep the column comparable from top to bottom.
    label = LABEL_MAP[emotion]
    print(f"\n=== alpha ladder: {emotion}, prompt {prompt_index} ===")
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
    payload = {
        "format_version": 1,
        "source": str(generations_path),
        "classifier_id": CLASSIFIER_ID,
        "similarity_id": SIMILARITY_ID if similarities is not None else None,
        "label_map": LABEL_MAP,
        "control_label": args.control_label,
        "response_count": len(scored),
        "emotions": emotions,
        "groups": summary,
    }
    (args.steering_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    header = (
        f"{'condition':>10} {'emotion':>12} {'alpha':>6} {'p(target)':>10} "
        f"{'delta':>8} {'up rate':>8} {'top=tgt':>8} {'sim':>6} {'rep3':>6} {'cjk':>6}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['condition']:>10} {row['emotion']:>12} {row['alpha']:>6.2f} "
            f"{row['target_probability']:>10.3f} "
            f"{(row['target_probability_delta'] or 0.0):>+8.3f} "
            f"{(row['increase_rate'] or 0.0):>8.2f} "
            f"{row['top_label_is_target_rate']:>8.2f} "
            f"{(row['baseline_similarity'] or 0.0):>6.2f} "
            f"{row['repeated_trigram_rate']:>6.2f} "
            f"{row['cjk_share']:>6.2f}"
        )

    if args.show_prompt is not None:
        _print_ladder(scored, args.show_prompt, args.show_emotion)

    if not args.no_figures:
        figure_dir = args.steering_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        try:
            _plot_dose_response(
                summary, emotions, figure_dir / "steering_dose_response.png"
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
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
