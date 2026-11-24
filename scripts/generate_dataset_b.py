from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from emotion_vectors import MODEL_ID, THINKING_ENABLED
from emotion_vectors.config import GENERATION_KWARGS
from emotion_vectors.conversations import (
    CONDITIONS,
    DEFAULT_CONDITIONS,
    SWITCH_TURN,
    build_dataset_b_manifest,
)
from emotion_vectors.generation import generate_chat_turns_batch, load_model_and_tokenizer

DEFAULT_RUN_DIR = Path("data/runs/dataset_b")

# Openings that mean the model acknowledged the mid-conversation instruction
# despite being told not to. The acknowledgement would dominate both the judge's
# rating and the first chunk of the pooled activations, so it has to be visible.
_ACKNOWLEDGEMENT = re.compile(
    r"^\W*(of course|certainly|sure|understood|noted|absolutely|got it|"
    r"i'?ll keep|i will keep|i'?ll stay|i will stay|as requested|happy to)",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows and make them durable before returning.

    Holding one file handle open across a long run loses everything when the
    process dies, because `flush()` only moves bytes from Python's buffer into
    the OS. On a Google Drive FUSE mount they then sit in a local cache that is
    uploaded on close or on a periodic sync, so a Colab disconnect discards the
    lot. `fsync` followed by closing the handle each time is what actually gets
    the batch to Drive. The cost is negligible beside a GPU generation step.
    """
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() and _read_jsonl(path) != rows:
        raise RuntimeError(
            f"existing manifest does not match the current configuration: {path}"
        )
    if not path.exists():
        _write_jsonl(path, rows)


def _quality_warnings(response: str, condition: str, *, meta_active: bool) -> list[str]:
    """Flag generations worth inspecting. Nothing is discarded automatically.

    `meta_active` is true from the switch turn onward in any arm that carries a
    mid-conversation meta instruction.
    """
    warnings = []
    lowered = response.lower()
    if re.search(r"[\u3400-\u9fff]", response):
        warnings.append("possible_language_switch")
    normalized_lines = [line.strip().lower() for line in response.splitlines() if line.strip()]
    if len(normalized_lines) != len(set(normalized_lines)):
        warnings.append("repeated_line")
    if condition == "suppression" and re.search(
        r"\b(calm|composed|professional)\b", lowered
    ):
        warnings.append("suppression_instruction_word_echo")
    if meta_active and _ACKNOWLEDGEMENT.match(response):
        warnings.append("meta_instruction_acknowledged")
    return warnings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paired natural and suppression Dataset B conversations."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float32",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--conversations-per-emotion", type=int, default=20)
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Generate both conditions for only the first N base conversations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Independent conversations generated together at each turn "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        choices=sorted(CONDITIONS),
        help=(
            "Arms to generate. The original run is 'natural suppression'; the "
            "mid-conversation experiment is 'natural_placebo switch'."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override the generation token budget. The default of "
            f"{GENERATION_KWARGS['max_new_tokens']} truncated most natural replies."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the sampling temperature.",
    )
    parser.add_argument("--thinking", action="store_true", default=THINKING_ENABLED)
    return parser


def main() -> None:
    args = _parser().parse_args()
    # generation.py holds a reference to this same dict and reads it at call
    # time, so an in-place override also lands in each row's
    # `generation_parameters` record.
    if args.max_new_tokens is not None:
        GENERATION_KWARGS["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        GENERATION_KWARGS["temperature"] = args.temperature

    manifest = build_dataset_b_manifest(
        conversations_per_emotion=args.conversations_per_emotion,
        conditions=args.conditions,
    )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.limit_conversations is not None:
        if args.limit_conversations < 1:
            raise SystemExit("--limit-conversations must be at least 1")
        selected_ids = list(
            dict.fromkeys(row["conversation_id"] for row in manifest)
        )[: args.limit_conversations]
        selected = set(selected_ids)
        selected_manifest = [
            row for row in manifest if row["conversation_id"] in selected
        ]
    else:
        selected_manifest = manifest

    args.run_dir.mkdir(parents=True, exist_ok=True)
    _prepare_manifest(args.run_dir / "manifest.jsonl", manifest)
    generation_path = args.run_dir / "generations.jsonl"
    existing = _read_jsonl(generation_path)
    by_sample_id = {row["sample_id"]: row for row in existing}
    expected_selected = len(selected_manifest) * 5
    if sum(
        f"{run['run_id']}_t{turn}" in by_sample_id
        for run in selected_manifest
        for turn in range(1, 6)
    ) == expected_selected:
        print(f"nothing to generate; all {expected_selected} selected responses are complete")
        return

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    completed_now = 0
    histories = {
        run["run_id"]: [{"role": "system", "content": run["instruction"]}]
        for run in selected_manifest
    }
    for turn in range(1, 6):
            for run in selected_manifest:
                histories[run["run_id"]].append(
                    {"role": "user", "content": run["user_turns"][turn - 1]}
                )

            for condition in dict.fromkeys(
                run["condition"] for run in selected_manifest
            ):
                condition_runs = [
                    run
                    for run in selected_manifest
                    if run["condition"] == condition
                ]
                for start in range(0, len(condition_runs), args.batch_size):
                    selected_batch = condition_runs[start : start + args.batch_size]
                    pending_runs = []
                    rows = []
                    for run in selected_batch:
                        sample_id = f"{run['run_id']}_t{turn}"
                        if sample_id in by_sample_id:
                            histories[run["run_id"]].append(
                                {
                                    "role": "assistant",
                                    "content": by_sample_id[sample_id]["response"],
                                }
                            )
                            continue
                        pending_runs.append(run)
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "run_id": run["run_id"],
                                "conversation_id": run["conversation_id"],
                                "emotion": run["emotion"],
                                "condition": run["condition"],
                                "scenario_index": run["scenario_index"],
                                "subject": run["subject"],
                                "turn": turn,
                                "user_text": run["user_turns"][turn - 1],
                                "meta_instruction": run.get("meta_instruction"),
                                "switch_turn": run.get("switch_turn"),
                                "meta_active": bool(run.get("switch_turn"))
                                and turn >= SWITCH_TURN,
                            }
                        )
                    if not pending_runs:
                        continue

                    batch_seed = selected_batch[0]["turn_seeds"][turn - 1]
                    records = generate_chat_turns_batch(
                        model,
                        tokenizer,
                        [histories[run["run_id"]] for run in pending_runs],
                        rows,
                        seed=batch_seed,
                        model_id=args.model_id,
                        thinking_enabled=args.thinking,
                    )
                    for run, record in zip(pending_runs, records, strict=True):
                        record["validation"]["warnings"].extend(
                            _quality_warnings(
                                record["response"],
                                condition,
                                meta_active=bool(record.get("meta_active")),
                            )
                        )
                        by_sample_id[record["sample_id"]] = record
                        histories[run["run_id"]].append(
                            {"role": "assistant", "content": record["response"]}
                        )
                        completed_now += 1
                        print(
                            f"generated {record['sample_id']} "
                            f"(valid={record['validation']['valid']})"
                        )
                    # Durable before the next batch, so a disconnect costs
                    # at most the batch in flight.
                    _append_jsonl(generation_path, records)

    print(f"generated {completed_now} responses; saved to {generation_path}")


if __name__ == "__main__":
    main()
