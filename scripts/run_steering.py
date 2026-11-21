"""PASS 3: generate steered responses across several layers and save them as JSONL.

Needs a GPU. For every (layer, emotion, alpha, prompt) combination this writes
one row, plus one unsteered baseline row per prompt and a same-norm
random-vector control row per (layer, alpha, prompt). Scoring happens later and
offline in `scripts/score_steering.py`.

Rows are generated baselines first, then one layer at a time, so that an
interrupted run leaves whole layers finished rather than every layer partial.

Example:
    uv run python scripts/run_steering.py \
      --run-dir data/runs/qwen35_2b_fp32 \
      --vector-dir data/frozen_vectors \
      --layers 6 12 16 23 \
      --torch-dtype float32 --device-map cuda
"""

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
DEFAULT_LAYERS = (6, 12, 16, 23)
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


def _row_id(
    condition: str,
    layer: int | None,
    emotion: str | None,
    alpha: float,
    prompt_index: int,
) -> str:
    """Stable identity of one generation, used for resuming.

    A baseline is unsteered, so it does not depend on the layer and is
    generated once and shared by every layer.
    """
    if condition == "baseline":
        return f"baseline_p{prompt_index:02d}"
    label = emotion if condition == "emotion" else condition
    return f"L{layer}_{label}_a{alpha:g}_p{prompt_index:02d}"


def run(args: argparse.Namespace) -> None:
    emotions = list(args.emotions)
    alphas = list(args.alphas)
    layers = list(dict.fromkeys(args.layers))
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
    layer_count = len(steering.decoder_layers(model))
    bad = [layer for layer in layers if not 0 <= layer < layer_count]
    if bad:
        raise ValueError(f"layers must be in [0, {layer_count - 1}], received {bad}")

    # The steering strength is a fraction of the layer's own typical token
    # norm, so each layer needs its own measurement.
    token_norms: dict[int, float] = {}
    zero_alpha_ok: dict[int, bool] = {}
    for layer in layers:
        token_norms[layer] = steering.mean_token_norm(
            model, records, layer, sample_size=args.norm_sample_size
        )
        print(
            f"layer {layer:>2}: typical per-token residual norm "
            f"{token_norms[layer]:.2f}"
        )

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "thinking_enabled": False,
    }

    for layer in layers:
        ok = steering.check_zero_alpha(
            model,
            tokenizer,
            prompts[0],
            layer=layer,
            hidden_size=hidden_size,
            **generate_kwargs,
        )
        zero_alpha_ok[layer] = ok
        print(f"layer {layer:>2}: alpha=0 reproduces the unsteered output: {ok}")
        if not ok and not args.allow_zero_alpha_mismatch:
            raise RuntimeError(
                f"a zero steering vector changed the output at layer {layer}, so "
                "the hook is wrong; pass --allow-zero-alpha-mismatch only if you "
                "understand why"
            )

    config = {
        "model_id": args.model_id,
        "torch_dtype": args.torch_dtype,
        "layers": layers,
        "mean_token_norm_per_layer": token_norms,
        "norm_sample_size": args.norm_sample_size,
        "emotions": emotions,
        "alphas": alphas,
        "prompts": prompts,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "seed": args.seed,
        "vector_source": str(args.vector_dir),
        "zero_alpha_check_passed": zero_alpha_ok,
        "steering_rule": "hidden_state += alpha * mean_token_norm[layer] * unit_vector",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    # (condition, layer, emotion, alpha, prompt_index)
    planned: list[tuple[str, int | None, str | None, float, int]] = [
        ("baseline", None, None, 0.0, index) for index in range(len(prompts))
    ]
    for layer in layers:
        for prompt_index in range(len(prompts)):
            for alpha in alphas:
                for emotion in emotions:
                    planned.append(("emotion", layer, emotion, alpha, prompt_index))
                planned.append(("random", layer, None, alpha, prompt_index))

    remaining = [item for item in planned if _row_id(*item) not in done]
    print(
        f"\n{len(planned)} rows planned over {len(layers)} layers, "
        f"{len(remaining)} still to generate"
    )

    with output_path.open("a", encoding="utf-8") as handle:
        for position, (condition, layer, emotion, alpha, prompt_index) in enumerate(
            remaining, start=1
        ):
            prompt = prompts[prompt_index]
            if condition == "baseline":
                vector = None
                token_norm = None
                hook_layer = layers[0]  # unused, no vector is applied
            elif condition == "emotion":
                token_norm = token_norms[layer]
                hook_layer = layer
                vector = steering.steering_vector(
                    unit_vectors[emotion][layer], alpha, token_norm, device=device
                )
            else:
                token_norm = token_norms[layer]
                hook_layer = layer
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
                layer=hook_layer,
                vector=vector,
                seed=args.seed,
                **generate_kwargs,
            )
            row = {
                "row_id": _row_id(condition, layer, emotion, alpha, prompt_index),
                "condition": condition,
                "layer": layer,
                "emotion": emotion,
                "alpha": alpha,
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
                where = "baseline" if layer is None else f"layer {layer}"
                print(f"  {position}/{len(remaining)} generated ({where})")

    print(f"\nsaved steering outputs to {output_path}")
    print(f"next: uv run python scripts/score_steering.py --steering-dir {output_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate steered responses across one or more layers."
    )
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
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="Decoder layers to steer at, one run each.",
    )
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
