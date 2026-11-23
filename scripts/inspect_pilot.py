"""Pilot inspection for the mid-conversation switch arm.

Answers the five questions that decide whether the switch design is viable,
using only `generations.jsonl`. No model, no judge, no GPU.

  1. Pipeline check   turns 1-3 must be identical between the two arms, since
                      they share a system prompt, user turns and seeds.
  2. Compliance       does the switch arm visibly flatten at turns 4-5?
  3. Acknowledgement  did the model answer the meta instruction despite being
                      told not to?
  4. Token budget     what fraction still hit the cap.
  5. Placebo inertness  does the control arm itself change at turn 4? If it
                      does, the placebo is not inert and needs replacing.

The final section prints turn-4 replies side by side, because at pilot stage
compliance is a judgement you make by reading, not by a metric.

Usage:
    uv run python scripts/inspect_pilot.py --run-dir data/runs/dataset_b_pilot
    uv run python scripts/inspect_pilot.py --run-dir ... --show 3 --chars 700
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CONTROL = "natural_placebo"
TREATMENT = "switch"
SWITCH_TURN = 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"no generations at {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def section(number: int, title: str) -> None:
    print(f"\n{'=' * 76}\n{number}. {title}\n{'=' * 76}")


def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.run_dir / "generations.jsonl")
    for r in rows:
        r["words"] = len(str(r.get("response", "")).split())
        r["warnings"] = list((r.get("validation") or {}).get("warnings") or [])

    conditions = sorted({r["condition"] for r in rows})
    print(f"{len(rows)} replies, conditions: {', '.join(conditions)}")
    if CONTROL not in conditions or TREATMENT not in conditions:
        print(
            f"\nThis script expects the '{CONTROL}' and '{TREATMENT}' arms.\n"
            f"Generate them with:\n"
            f"  --conditions {CONTROL} {TREATMENT}"
        )
        return

    indexed: dict[tuple[str, str, int], dict[str, Any]] = {
        (r["conversation_id"], r["condition"], int(r["turn"])): r for r in rows
    }
    conversations = sorted({r["conversation_id"] for r in rows})

    # ----------------------------------------------------------------- 1
    section(1, "PIPELINE CHECK  (turns 1-3 must be identical across arms)")
    mismatches = []
    compared = 0
    for conversation in conversations:
        for turn in range(1, SWITCH_TURN):
            a = indexed.get((conversation, CONTROL, turn))
            b = indexed.get((conversation, TREATMENT, turn))
            if a is None or b is None:
                continue
            compared += 1
            if a["response"] != b["response"]:
                mismatches.append((conversation, turn))
    print(f"  compared {compared} reply pairs across turns 1-{SWITCH_TURN - 1}")
    if not mismatches:
        print("  OK: byte-identical. The pairing holds and batching is deterministic.")
    else:
        print(f"  {len(mismatches)} MISMATCHES, e.g. {mismatches[:5]}")
        print(
            "  The arms diverge before the switch, so differences at turns 4-5\n"
            "  cannot be attributed to the instruction. Likely causes: batch\n"
            "  composition differs between arms, or seeds are not shared."
        )

    # ----------------------------------------------------------------- 2
    section(2, "COMPLIANCE  (reply length by turn and arm)")
    print(f"{'turn':>6}", end="")
    for condition in (CONTROL, TREATMENT):
        print(f"{condition:>20}", end="")
    print(f"{'difference':>13}")
    for turn in sorted({int(r["turn"]) for r in rows}):
        marker = " <- switch" if turn == SWITCH_TURN else ""
        control = [
            indexed[(c, CONTROL, turn)]["words"]
            for c in conversations
            if (c, CONTROL, turn) in indexed
        ]
        treatment = [
            indexed[(c, TREATMENT, turn)]["words"]
            for c in conversations
            if (c, TREATMENT, turn) in indexed
        ]
        print(
            f"{turn:>6}{mean(control):>20.1f}{mean(treatment):>20.1f}"
            f"{mean(treatment) - mean(control):>+13.1f}{marker}"
        )
    print(
        "\n  A clear negative difference from turn 4 onward is the first sign the\n"
        "  instruction landed. No change at all means the model ignored it."
    )

    # ----------------------------------------------------------------- 3
    section(3, "ACKNOWLEDGEMENT LEAK  (told not to acknowledge; did it anyway?)")
    for condition in (CONTROL, TREATMENT):
        post = [
            r
            for r in rows
            if r["condition"] == condition and int(r["turn"]) >= SWITCH_TURN
        ]
        leaked = [r for r in post if "meta_instruction_acknowledged" in r["warnings"]]
        rate = len(leaked) / len(post) if post else float("nan")
        print(f"  {condition:>18}  {len(leaked)}/{len(post)} replies  ({rate:.0%})")
    print(
        "\n  A high rate means the opening sentence is compliance talk rather than\n"
        "  content, which contaminates both the judge rating and the pooled\n"
        "  activations. Fix by strengthening the wording, not by stripping text."
    )

    # ----------------------------------------------------------------- 4
    section(4, "TOKEN BUDGET")
    budget = None
    params = rows[0].get("generation_parameters") or {}
    if "max_new_tokens" in params:
        budget = int(params["max_new_tokens"])
    print(f"  max_new_tokens = {budget}")
    for condition in (CONTROL, TREATMENT):
        subset = [r for r in rows if r["condition"] == condition]
        truncated = [r for r in subset if "possibly_truncated" in r["warnings"]]
        lengths = sorted(r["words"] for r in subset)
        print(
            f"  {condition:>18}  truncated {len(truncated)}/{len(subset)} "
            f"({len(truncated) / len(subset):.0%})   "
            f"median {lengths[len(lengths) // 2]} words   max {lengths[-1]}"
        )
    print(
        "\n  Aim for a low rate in BOTH arms. An asymmetric rate reintroduces the\n"
        "  confound that length, not emotion, drives the measured difference."
    )

    # ----------------------------------------------------------------- 5
    section(5, "PLACEBO INERTNESS  (does the control arm change at the switch?)")
    before = [
        indexed[(c, CONTROL, SWITCH_TURN - 1)]["words"]
        for c in conversations
        if (c, CONTROL, SWITCH_TURN - 1) in indexed
    ]
    after = [
        indexed[(c, CONTROL, SWITCH_TURN)]["words"]
        for c in conversations
        if (c, CONTROL, SWITCH_TURN) in indexed
    ]
    print(
        f"  control arm, turn {SWITCH_TURN - 1}: {mean(before):.1f} words  ->  "
        f"turn {SWITCH_TURN}: {mean(after):.1f} words  "
        f"({mean(after) - mean(before):+.1f})"
    )
    print(
        "  Length alone cannot settle this; the real test is the emotion score,\n"
        "  which needs the scoring pass. If the control arm also flattens, the\n"
        "  placebo is doing work and should move to an orthogonal style\n"
        "  dimension instead of 'clear and helpful'."
    )

    # ----------------------------------------------------------------- 6
    section(6, f"READ THESE  (turn {SWITCH_TURN}, control vs switch)")
    shown = 0
    by_emotion: dict[str, list[str]] = defaultdict(list)
    for c in conversations:
        by_emotion[c.rsplit("_", 1)[0]].append(c)
    for emotion in sorted(by_emotion):
        if shown >= args.show:
            break
        conversation = by_emotion[emotion][0]
        a = indexed.get((conversation, CONTROL, SWITCH_TURN))
        b = indexed.get((conversation, TREATMENT, SWITCH_TURN))
        if a is None or b is None:
            continue
        shown += 1
        print(f"\n  --- {conversation} ---")
        print(f"  user: {a['user_text'][:160]}")
        for label, record in ((CONTROL, a), (TREATMENT, b)):
            text = " ".join(str(record["response"]).split())
            print(f"\n  [{label}] ({record['words']} words)")
            print(f"    {text[:args.chars]}")
    print(
        "\n  Compliance at pilot stage is a judgement you make by reading. If the\n"
        "  switch replies are visibly flatter, proceed to the full run."
    )

    report_internal(args.run_dir)


def report_internal(run_dir: Path) -> None:
    """Section 7: the emotion scores, if internal scoring has been run.

    Uses the per-reply centered score, `target minus the mean of all ten
    projections on the same reply`. That needs no cross-reply estimation, so it
    works at pilot sample sizes. It still carries the per-emotion constant
    described in analyze_theater.py section 3, but that constant is identical
    in both arms and at every turn, so it cancels in the arm-against-arm and
    turn-against-turn comparisons that matter here.

    Two questions this answers that reading cannot:
      - does the control arm still carry emotion after the length cap?
      - does the switch arm's INTERNAL emotion drop at turns 4-5, or only its
        wording?
    """
    path = run_dir / "internal_scores.jsonl"
    if not path.exists():
        print(
            f"\n{'=' * 76}\n7. EMOTION SCORES\n{'=' * 76}\n"
            f"  not scored yet. Run:\n"
            f"    python scripts/score_internal_b.py --run-dir {run_dir} \\\n"
            f"      --vector-dir <frozen vectors> --torch-dtype float32"
        )
        return

    scores = read_jsonl(path)
    emotions = list(scores[0]["projections"].keys())
    centered: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in scores:
        projections = {e: float(record["projections"][e]) for e in emotions}
        value = projections[record["emotion"]] - sum(projections.values()) / len(emotions)
        centered[(record["condition"], int(record["turn"]))].append(value)

    section(7, "EMOTION SCORES  (per-reply centered target projection)")
    print(f"{'turn':>6}{CONTROL:>20}{TREATMENT:>20}{'difference':>13}")
    for turn in sorted({t for _, t in centered}):
        control = centered.get((CONTROL, turn), [])
        treatment = centered.get((TREATMENT, turn), [])
        if not control or not treatment:
            continue
        marker = " <- switch" if turn == SWITCH_TURN else ""
        print(
            f"{turn:>6}{mean(control):>20.4f}{mean(treatment):>20.4f}"
            f"{mean(treatment) - mean(control):>+13.4f}{marker}"
        )

    pre = [v for t in range(1, SWITCH_TURN) for v in centered.get((CONTROL, t), [])]
    post = centered.get((CONTROL, SWITCH_TURN), [])
    print(
        f"\n  Control arm, turns 1-{SWITCH_TURN - 1}: {mean(pre):+.4f}  ->  "
        f"turn {SWITCH_TURN}: {mean(post):+.4f}"
    )
    print(
        "  A control value near zero means the length cap flattened the control\n"
        "  arm too, leaving nothing for the switch arm to suppress. A clearly\n"
        "  positive control value with a negative difference at turns 4-5 is the\n"
        "  result the experiment is looking for."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a switch-arm pilot run.")
    parser.add_argument("--run-dir", type=Path, default=Path("data/runs/dataset_b_pilot"))
    parser.add_argument("--show", type=int, default=3, help="Conversations to print.")
    parser.add_argument("--chars", type=int, default=600, help="Characters per reply.")
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
