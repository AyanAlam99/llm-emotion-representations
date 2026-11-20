
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from emotion_vectors import steering
from emotion_vectors.config import MODEL_ID, SEED
from emotion_vectors.generation import load_model_and_tokenizer

DEFAULT_RUN_DIR = Path("data/runs/qwen35_2b_fp32")
DEFAULT_LAYER = 12
DEFAULT_EMOTIONS = ("joy", "sadness", "anger", "fear")
DEFAULT_ALPHAS = (0.25, 0.5, 1.0, 2.0)

# Emotionally flat, everyday prompts. If a steered response turns emotional,
# the emotion came from the vector and not from the prompt.
NEUTRAL_PROMPTS: tuple[str, ...] = (
    "Describe your morning routine.",
    "Explain how to make a cup of tea.",
    "Write a short paragraph about a walk to the bus stop.",
    "Describe what a public library looks like inside.",
    "Explain what happens during a routine dentist check-up.",
    "Write a short paragraph about doing the weekly grocery shopping.",
    "Describe the process of washing dishes by hand.",
    "Explain how to pack a suitcase for a three-day trip.",
    "Write a short paragraph about waiting for a train.",
    "Describe what you would see on a quiet street at noon.",
    "Explain how to water indoor plants.",
    "Write a short paragraph about organising a desk.",
)

CJK_PATTERN = re.compile(r"[一-鿿぀-ヿ가-힯]")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_unit_vectors(vector_dir: Path, emotions: list[str]) -> dict[str, torch.Tensor]:
    path = vector_dir / "emotion_vectors.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"vector artifact does not exist: {path}")
    tensors = load_file(path, device="cpu")
    missing = [
        emotion
        for emotion in emotions
        if f"unit/{emotion}/all_layers" not in tensors
    ]
    if missing:
        raise KeyError(f"vector file has no unit vectors for: {missing}")
    return {
        emotion: tensors[f"unit/{emotion}/all_layers"].float() for emotion in emotions
    }


def text_metrics(text: str) -> dict[str, Any]:
    """Cheap degeneration checks that need no extra model.

    `repeated_trigram_rate` is the share of 3-word windows that are not unique;
    it rises sharply once steering pushes the model into looping. `cjk_share`
    catches the Chinese-token drift Jeong reported for Qwen at high strength.
    """
    words = text.split()
    trigrams = [tuple(words[i : i + 3]) for i in range(max(len(words) - 2, 0))]
    unique = len(set(trigrams))
    repeated_rate = 1.0 - (unique / len(trigrams)) if trigrams else 0.0
    characters = [character for character in text if not character.isspace()]
    cjk = sum(1 for character in characters if CJK_PATTERN.match(character))
    return {
        "word_count": len(words),
        "character_count": len(text),
        "repeated_trigram_rate": round(repeated_rate, 4),
        "cjk_share": round(cjk / len(characters), 4) if characters else 0.0,
    }


def _row_id(condition: str, emotion: str | None, alpha: float, prompt_index: int) -> str:
    label = emotion if condition == "emotion" else condition
    return f"{label}_a{alpha:g}_p{prompt_index:02d}"


def run(args: argparse.Namespace) -> None:
    emotions = list(args.emotions)
    alphas = list(args.alphas)
    prompts = list(NEUTRAL_PROMPTS)[: args.prompt_count]

    records = _read_jsonl(args.run_dir / "generations.jsonl")
    unit_vectors = _load_unit_vectors(args.vector_dir, emotions)

    output_dir = args.output_dir or (args.run_dir / "steering")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generations.jsonl"

    done: set[str] = set()
    if output_path.exists() and not args.overwrite:
        done = {row["row_id"] for row in _read_jsonl(output_path)}
        print(f"resuming: {len(done)} rows already present in {output_path}")
    elif output_path.exists():
        output_path.unlink()

    print(f"loading {args.model_id} as {args.torch_dtype}...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, torch_dtype=args.torch_dtype, device_map=args.device_map
    )
    device = steering._input_device(model)
    hidden_size = model.config.hidden_size

    token_norm = steering.mean_token_norm(
        model, records, args.layer, sample_size=args.norm_sample_size
    )
    print(
        f"layer {args.layer}: typical per-token residual norm {token_norm:.2f} "
        f"(measured on {args.norm_sample_size} neutral responses)"
    )

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "thinking_enabled": False,
    }

    ok = steering.check_zero_alpha(
        model,
        tokenizer,
        prompts[0],
        layer=args.layer,
        hidden_size=hidden_size,
        **generate_kwargs,
    )
    print(f"alpha=0 reproduces the unsteered output: {ok}")
    if not ok and not args.allow_zero_alpha_mismatch:
        raise RuntimeError(
            "a zero steering vector changed the output, so the hook is wrong; "
            "pass --allow-zero-alpha-mismatch only if you understand why"
        )

    config = {
        "model_id": args.model_id,
        "torch_dtype": args.torch_dtype,
        "layer": args.layer,
        "mean_token_norm": token_norm,
        "norm_sample_size": args.norm_sample_size,
        "emotions": emotions,
        "alphas": alphas,
        "prompts": prompts,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "seed": args.seed,
        "vector_source": str(args.vector_dir),
        "zero_alpha_check_passed": ok,
        "steering_rule": "hidden_state += alpha * mean_token_norm * unit_vector",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    planned: list[tuple[str, str | None, float, int]] = []
    for prompt_index in range(len(prompts)):
        planned.append(("baseline", None, 0.0, prompt_index))
        for alpha in alphas:
            for emotion in emotions:
                planned.append(("emotion", emotion, alpha, prompt_index))
            planned.append(("random", None, alpha, prompt_index))

    remaining = [item for item in planned if _row_id(*item) not in done]
    print(f"{len(planned)} rows planned, {len(remaining)} to generate")

    with output_path.open("a", encoding="utf-8") as handle:
        for position, (condition, emotion, alpha, prompt_index) in enumerate(
            remaining, start=1
        ):
            prompt = prompts[prompt_index]
            if condition == "baseline":
                vector = None
            elif condition == "emotion":
                vector = steering.steering_vector(
                    unit_vectors[emotion][args.layer],
                    alpha,
                    token_norm,
                    device=device,
                )
            else:
                vector = steering.random_control_vector(
                    hidden_size,
                    alpha,
                    token_norm,
                    seed=args.seed + prompt_index,
                    device=device,
                )

            response = steering.generate_steered(
                model,
                tokenizer,
                prompt,
                layer=args.layer,
                vector=vector,
                seed=args.seed,
                **generate_kwargs,
            )
            row = {
                "row_id": _row_id(condition, emotion, alpha, prompt_index),
                "condition": condition,
                "emotion": emotion,
                "alpha": alpha,
                "layer": args.layer,
                "mean_token_norm": token_norm,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "response": response,
                "metrics": text_metrics(response),
                "model_id": args.model_id,
                "seed": args.seed,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 10 == 0 or position == len(remaining):
                print(f"  {position}/{len(remaining)} generated")

    print(f"\nsaved steering outputs to {output_path}")
    print("next: uv run python scripts/score_steering.py --steering-dir " f"{output_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate steered responses.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=Path("data/frozen_vectors"),
        help="Directory with emotion_vectors.safetensors.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--emotions", nargs="*", default=list(DEFAULT_EMOTIONS))
    parser.add_argument("--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS))
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=len(NEUTRAL_PROMPTS),
        help="How many of the built-in neutral prompts to use.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--norm-sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-zero-alpha-mismatch",
        action="store_true",
        help="Continue even if a zero steering vector changes the output.",
    )
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
