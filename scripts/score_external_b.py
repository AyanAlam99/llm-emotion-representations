from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from emotion_vectors.config import SEED
from emotion_vectors.external_judge import blind, run_claude

DEFAULT_RUN_DIR = Path("data/runs/dataset_b")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_or_check(path: Path, rows: list[dict]) -> None:
    if path.exists() and read_jsonl(path) != rows:
        raise RuntimeError(f"existing blinded data does not match: {path}")
    if not path.exists():
        write_jsonl(path, rows)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Blindly rate Dataset B responses with logged-in Claude Code."
    )
    result.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    result.add_argument("--model", default="claude-haiku-5-5")
    result.add_argument("--effort", default="low")
    result.add_argument("--batch-size", type=int, default=20)
    result.add_argument("--limit", type=int, default=None)
    result.add_argument("--seed", type=int, default=SEED)
    result.add_argument("--max-retries", type=int, default=3)
    result.add_argument("--timeout", type=int, default=900)
    result.add_argument("--claude", default="claude")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.batch_size < 1 or args.max_retries < 1:
        raise SystemExit("--batch-size and --max-retries must be positive")

    generations = read_jsonl(args.run_dir / "generations.jsonl")
    if not generations:
        raise SystemExit("Dataset B generations are missing")
    all_requests, mapping = blind(generations, args.seed)
    requests = all_requests
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        requests = requests[: args.limit]

    judge_dir = args.run_dir / "judge"
    transcript_dir = judge_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    write_or_check(judge_dir / "blind_requests.jsonl", all_requests)
    map_rows = [
        {"response_id": response_id, "sample_id": sample_id}
        for response_id, sample_id in mapping.items()
    ]
    write_or_check(judge_dir / "blind_map.jsonl", map_rows)

    output_path = args.run_dir / "external_scores.jsonl"
    existing = read_jsonl(output_path)
    if any(
        row.get("judge_model") != args.model or row.get("judge_effort") != args.effort
        for row in existing
    ):
        raise RuntimeError("existing scores use a different judge model or effort")
    completed = {row["response_id"] for row in existing}

    unjudged_path = judge_dir / "unjudged.jsonl"

    def attempt_batch(batch: list[dict], label: str):
        """Judge one batch, retrying. Returns None when it never succeeds."""
        for attempt in range(1, args.max_retries + 1):
            try:
                return run_claude(
                    batch,
                    model=args.model,
                    effort=args.effort,
                    executable=args.claude,
                    timeout=args.timeout,
                )
            except (RuntimeError, ValueError) as exc:
                if attempt == args.max_retries:
                    print(f"{label} failed after {args.max_retries} attempts: {exc}")
                    return None
                print(f"{label} failed: {exc}; retrying")
                time.sleep(2**attempt)
        return None

    with output_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(requests), args.batch_size):
            batch = [
                row
                for row in requests[start : start + args.batch_size]
                if row["response_id"] not in completed
            ]
            if not batch:
                continue
            batch_number = start // args.batch_size + 1

            result = attempt_batch(batch, f"batch {batch_number}")
            # The judge can decline a whole batch over the content of a single
            # reply, which used to abort the run. Fall back to judging the batch
            # one reply at a time so the loss is isolated to the offending one,
            # and record it rather than dropping it silently. `load_joined` in
            # analyze_theater.py already skips replies with no judge row.
            groups = [(result, batch)]
            if result is None and len(batch) > 1:
                print(f"  splitting batch {batch_number} into {len(batch)} singles")
                groups = [
                    (attempt_batch([row], f"  {row['response_id']}"), [row])
                    for row in batch
                ]

            for outcome, group in groups:
                if outcome is None:
                    for row in group:
                        record = {
                            "response_id": row["response_id"],
                            "sample_id": mapping[row["response_id"]],
                            "reason": "judge declined or errored on every attempt",
                        }
                        with unjudged_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        print(f"  UNJUDGED {record['sample_id']}")
                    continue
                ratings, transcript, metadata = outcome
                transcript_path = transcript_dir / (
                    f"batch_{batch_number:04d}_{metadata['session_id']}.jsonl"
                )
                transcript_path.write_text(transcript, encoding="utf-8")
                for row in ratings:
                    record = {
                        "sample_id": mapping[row["response_id"]],
                        **row,
                        "judge_provider": "claude-code",
                        "judge_model": args.model,
                        "judge_effort": args.effort,
                        **metadata,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed.add(row["response_id"])
            output.flush()
            done = sum(row["response_id"] in completed for row in requests)
            print(f"judged {done}/{len(requests)} responses")

    print(f"saved external ratings to {output_path}")
    if unjudged_path.exists():
        skipped = len(read_jsonl(unjudged_path))
        print(f"{skipped} replies could not be judged; see {unjudged_path}")


if __name__ == "__main__":
    main()
