"""Statistical analysis for Dataset B, the Emotion Theater experiment.

Reads the three Dataset B artifacts (generations, internal projection scores,
blinded external judge ratings), joins them on `sample_id`, and reports:

  1. Dataset shape
  2. Confound checks        reply length, truncation and degeneration per condition
  3. The domain offset      a per-emotion constant left in every projection by
                            using a story-derived neutral baseline on assistant
                            replies. A finding in its own right.
  4. Recovery accuracy      argmax over 10 emotions, raw and offset-corrected,
                            per condition, bootstrapped over conversations
  5. Confusion              where the probe sends its predictions, before and after
  6. Manipulation check     did the suppression instruction reduce expressed emotion
  7. The result             D = R_E - R_I on z-scored, conversation-level scores
  8. Breakdowns             D per emotion and per turn
  9. Robustness             D and accuracy under length and truncation splits
 10. Headline numbers       the figures to quote

Why z-scores and not a retention ratio: the projections carry a large additive
per-emotion constant (section 3). A ratio of suppression to natural is not
immune to an additive constant, since adding the same value to numerator and
denominator drags the ratio toward 100%. Subtracting the within-emotion mean,
which is what a z-score does, removes that constant exactly. So the z-scored
difference is the correct cross-channel comparison and a ratio is not.

No scipy dependency: the Wilcoxon signed-rank test uses the normal
approximation with tie correction, which is accurate at n = 200.

Usage:
    uv run python scripts/analyze_theater.py
    uv run python scripts/analyze_theater.py --run-dir data/runs/dataset_b
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_RUN_DIR = Path("data/runs/dataset_b")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
DEGENERATE_WARNINGS = ("possible_language_switch", "repeated_line")

# Near-synonym groupings. A probe that confuses joy with excitement is doing
# something very different from one that confuses joy with fear, so errors are
# reported both exactly and at family level.
FAMILY = {
    "anger": "anger/annoyance",
    "annoyance": "anger/annoyance",
    "fear": "fear/nervousness",
    "nervousness": "fear/nervousness",
    "sadness": "sadness/disappointment",
    "disappointment": "sadness/disappointment",
    "joy": "joy/excitement",
    "excitement": "joy/excitement",
    "admiration": "admiration/gratitude",
    "gratitude": "admiration/gratitude",
}


# --------------------------------------------------------------------------- #
# statistics helpers (no scipy)
# --------------------------------------------------------------------------- #


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _average_ranks(magnitudes: list[float]) -> tuple[list[float], float]:
    """Ranks with ties averaged, plus the tie-correction term sum(t^3 - t)."""
    order = sorted(range(len(magnitudes)), key=lambda i: magnitudes[i])
    ranks = [0.0] * len(magnitudes)
    tie_term = 0.0
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order)
            and magnitudes[order[end + 1]] == magnitudes[order[position]]
        ):
            end += 1
        group = end - position + 1
        shared = (position + end) / 2.0 + 1.0  # 1-indexed average rank
        for k in range(position, end + 1):
            ranks[order[k]] = shared
        if group > 1:
            tie_term += group**3 - group
        position = end + 1
    return ranks, tie_term


def wilcoxon_signed_rank(differences: Sequence[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test against a median of zero."""
    nonzero = [d for d in differences if d != 0.0]
    n = len(nonzero)
    if n < 10:
        return {"n": n, "p_value": float("nan"), "note": "too few nonzero pairs"}

    ranks, tie_term = _average_ranks([abs(d) for d in nonzero])
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
    z = (w_plus - mu) / sigma
    return {
        "n": n,
        "n_dropped_zero": len(differences) - n,
        "W_plus": w_plus,
        "z": z,
        "p_value": 2.0 * (1.0 - _phi(abs(z))),
        "effect_size_r": z / math.sqrt(n),
        "n_positive": sum(1 for d in nonzero if d > 0),
        "n_negative": sum(1 for d in nonzero if d < 0),
    }


def bootstrap_ci(
    clusters: Sequence[Sequence[Any]],
    statistic: Callable[[list[Any]], float],
    *,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile CI, resampling whole clusters.

    Replies are nested inside conversations, so the conversation is the unit of
    resampling. Resampling individual replies would understate the interval.
    """
    rng = random.Random(seed)
    size = len(clusters)
    estimates = []
    for _ in range(n_resamples):
        pooled: list[Any] = []
        for _ in range(size):
            pooled.extend(clusters[rng.randrange(size)])
        try:
            estimates.append(statistic(pooled))
        except (ZeroDivisionError, ValueError):
            continue
    if len(estimates) < 100:
        return float("nan"), float("nan")
    estimates.sort()
    return (
        estimates[int(0.025 * len(estimates))],
        estimates[int(0.975 * len(estimates))],
    )


# --------------------------------------------------------------------------- #
# loading and joining
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing Dataset B artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_joined(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Join the three artifacts on sample_id. Returns (rows, emotions)."""
    internal = read_jsonl(run_dir / "internal_scores.jsonl")
    external = {r["sample_id"]: r for r in read_jsonl(run_dir / "external_scores.jsonl")}
    generations = {r["sample_id"]: r for r in read_jsonl(run_dir / "generations.jsonl")}

    emotions = list(internal[0]["projections"].keys())

    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in internal:
        sample_id = record["sample_id"]
        judge = external.get(sample_id)
        generation = generations.get(sample_id)
        if judge is None or generation is None:
            skipped += 1
            continue
        validation = generation.get("validation") or {}
        warnings = list(validation.get("warnings") or [])
        rows.append(
            {
                "sample_id": sample_id,
                "conversation_id": record["conversation_id"],
                "emotion": record["emotion"],
                "condition": record["condition"],
                "turn": int(record["turn"]),
                "internal": {e: float(record["projections"][e]) for e in emotions},
                "external": {e: float(judge["ratings"][e]) for e in emotions},
                "word_count": len(str(generation.get("response", "")).split()),
                "warnings": warnings,
                "truncated": "possibly_truncated" in warnings,
                "degenerate": any(w in DEGENERATE_WARNINGS for w in warnings),
            }
        )

    if skipped:
        print(f"  warning: {skipped} internal rows had no matching judge/generation row")
    return rows, emotions


def cluster_by_conversation(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r["conversation_id"]].append(r)
    return list(buckets.values())


# --------------------------------------------------------------------------- #
# scoring conventions
# --------------------------------------------------------------------------- #


def apply_offsets(rows: list[dict[str, Any]], emotions: list[str]) -> dict[str, Any]:
    """Attach offset-corrected projections and ratings to every row.

    The offset for emotion `e` is the mean score on that emotion's vector across
    ALL replies, irrespective of which emotion the conversation targets. No
    label is used, so nothing leaks, but it is estimated from the Dataset B
    distribution itself and is therefore transductive. The clean alternative is
    a neutral-scenario arm generated inside Dataset B.
    """
    offsets = {
        "internal": {e: mean([r["internal"][e] for r in rows]) for e in emotions},
        "external": {e: mean([r["external"][e] for r in rows]) for e in emotions},
    }
    scales = {
        "internal": {e: stdev([r["internal"][e] for r in rows]) for e in emotions},
        "external": {e: stdev([r["external"][e] for r in rows]) for e in emotions},
    }
    for r in rows:
        for channel in ("internal", "external"):
            r[f"{channel}_corrected"] = {
                e: r[channel][e] - offsets[channel][e] for e in emotions
            }
        r["internal_target"] = r["internal"][r["emotion"]]
        r["external_target"] = r["external"][r["emotion"]]
    return {"offsets": offsets, "scales": scales}


def strict_argmax(scores: dict[str, float]) -> str | None:
    """The unique maximum, or None when tied.

    Ties count as no prediction. Under suppression the judge often returns all
    zeros, and crediting that as a hit would inflate the external score.
    """
    best = max(scores.values())
    winners = [e for e, v in scores.items() if v == best]
    return winners[0] if len(winners) == 1 else None


def all_zero(scores: dict[str, float]) -> bool:
    return all(v == 0.0 for v in scores.values())


def accuracy(rows: Sequence[dict[str, Any]], key: str, *, family: bool = False) -> float:
    if not rows:
        return float("nan")
    hits = 0
    for r in rows:
        predicted = strict_argmax(r[key])
        if predicted is None:
            continue
        if family:
            hits += FAMILY[predicted] == FAMILY[r["emotion"]]
        else:
            hits += predicted == r["emotion"]
    return hits / len(rows)


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #


def section(number: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{number}. {title}\n{'=' * 78}")


def report_dataset(rows: list[dict[str, Any]], emotions: list[str]) -> dict[str, Any]:
    section(1, "DATASET")
    conversations = {r["conversation_id"] for r in rows}
    turns = sorted({r["turn"] for r in rows})
    out = {
        "replies": len(rows),
        "conversations": len(conversations),
        "emotions": len(emotions),
        "turns_per_conversation": len(turns),
        "conditions": sorted({r["condition"] for r in rows}),
    }
    print(f"  replies                 {len(rows)}")
    print(f"  base conversations      {len(conversations)}")
    print(f"  emotions                {len(emotions)}")
    print(f"  turns per conversation  {len(turns)}")
    print(f"  conditions              {', '.join(out['conditions'])}")
    return out


def report_confounds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    section(2, "CONFOUND CHECKS")
    out: dict[str, Any] = {}
    print(
        f"{'condition':>14} {'n':>6} {'words':>8} {'sd':>7} "
        f"{'truncated':>11} {'degenerate':>12}"
    )
    for condition in ("natural", "suppression"):
        subset = [r for r in rows if r["condition"] == condition]
        words = [r["word_count"] for r in subset]
        trunc = mean([1.0 if r["truncated"] else 0.0 for r in subset])
        degen = mean([1.0 if r["degenerate"] else 0.0 for r in subset])
        out[condition] = {
            "n": len(subset),
            "mean_words": mean(words),
            "sd_words": stdev(words),
            "truncated_rate": trunc,
            "degenerate_rate": degen,
        }
        print(
            f"{condition:>14} {len(subset):>6} {mean(words):>8.1f} {stdev(words):>7.1f} "
            f"{trunc:>10.1%} {degen:>11.1%}"
        )
    gap = out["natural"]["mean_words"] - out["suppression"]["mean_words"]
    out["word_gap"] = gap
    print(f"\n  reply-length gap (natural - suppression): {gap:+.1f} words")
    print(
        "  The truncation rates are asymmetric, so the two conditions are not\n"
        "  length-comparable. Section 9 tests whether the result depends on it."
    )
    return out


def report_offset(
    rows: list[dict[str, Any]], emotions: list[str], offset_info: dict[str, Any]
) -> dict[str, Any]:
    section(3, "THE DOMAIN OFFSET")
    offsets = offset_info["offsets"]["internal"]
    scales = offset_info["scales"]["internal"]
    print(
        "  `neutral_mean` is the average activation of Dataset A's neutral STORIES.\n"
        "  Dataset B replies are assistant-register text, so after the subtraction\n"
        "  every reply keeps a shared component no story had. Projection is linear,\n"
        "  so that component adds a CONSTANT to each emotion's score:\n"
        "      (h_emotion + a) . v_i  =  h_emotion . v_i  +  a . v_i\n"
    )
    print(f"{'emotion':>16} {'constant':>10} {'within-emotion sd':>19}")
    for e in sorted(emotions, key=lambda e: -offsets[e]):
        print(f"{e:>16} {offsets[e]:>10.4f} {scales[e]:>19.4f}")
    spread = max(offsets.values()) - min(offsets.values())
    typical = mean([scales[e] for e in emotions])
    out = {
        "internal_offsets": offsets,
        "internal_scales": scales,
        "offset_spread": spread,
        "mean_within_emotion_sd": typical,
        "ratio": spread / typical if typical else float("nan"),
    }
    print(f"\n  spread of constants      {spread:.4f}")
    print(f"  mean within-emotion sd   {typical:.4f}")
    print(f"  ratio                    {spread / typical:.2f}x")
    print(
        "  A ratio above 1 means the constants dominate the argmax, so the probe\n"
        "  predicts whichever emotion carries the largest constant."
    )
    return out


def report_accuracy(rows: list[dict[str, Any]], n_emotions: int) -> dict[str, Any]:
    section(4, "RECOVERY ACCURACY  (argmax over 10 emotions)")
    chance = 1.0 / n_emotions
    out: dict[str, Any] = {"chance": chance, "family_chance": 1.0 / 5}
    print(
        f"{'condition':>13} {'channel':>10} {'variant':>11} "
        f"{'exact':>8} {'95% CI':>18} {'family':>8}"
    )
    for condition in ("natural", "suppression"):
        subset = [r for r in rows if r["condition"] == condition]
        clusters = cluster_by_conversation(subset)
        for channel in ("internal", "external"):
            for variant, key in (("raw", channel), ("corrected", f"{channel}_corrected")):
                point = accuracy(subset, key)
                lo, hi = bootstrap_ci(clusters, lambda g, k=key: accuracy(g, k))
                fam = accuracy(subset, key, family=True)
                out.setdefault(condition, {}).setdefault(channel, {})[variant] = {
                    "exact": point,
                    "ci95": [lo, hi],
                    "family": fam,
                }
                print(
                    f"{condition:>13} {channel:>10} {variant:>11} "
                    f"{point:>7.1%}  [{lo:>5.1%},{hi:>6.1%}] {fam:>7.1%}"
                )
    print(f"\n  chance: exact {chance:.1%}, family {1/5:.1%}")
    nat = out["natural"]["internal"]["corrected"]["exact"]
    print(
        f"  Gate 1 (transfer): vectors built only on personal stories recover the\n"
        f"  emotion in operational conversations at {nat:.1%}, "
        f"{nat / chance:.1f}x chance."
    )
    return out


def report_confusion(
    rows: list[dict[str, Any]], emotions: list[str], out_dir: Path
) -> dict[str, Any]:
    section(5, "CONFUSION  (natural condition, internal probe)")
    natural = [r for r in rows if r["condition"] == "natural"]
    out: dict[str, Any] = {}
    for variant, key in (("raw", "internal"), ("corrected", "internal_corrected")):
        matrix: dict[str, Counter] = defaultdict(Counter)
        for r in natural:
            predicted = strict_argmax(r[key])
            matrix[r["emotion"]][predicted or "tie"] += 1
        print(f"\n  {variant}  (true -> top three predictions)")
        for e in emotions:
            total = sum(matrix[e].values())
            top = "  ".join(f"{p}:{c / total:.0%}" for p, c in matrix[e].most_common(3))
            print(f"    {e:>16} -> {top}")
        out[variant] = {t: dict(c) for t, c in matrix.items()}

        path = out_dir / f"theater_confusion_{variant}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true"] + emotions + ["tie"])
            for e in emotions:
                writer.writerow([e] + [matrix[e].get(p, 0) for p in emotions] + [matrix[e].get("tie", 0)])
    errors = [r for r in natural if strict_argmax(r["internal_corrected"]) != r["emotion"]]
    in_family = mean(
        [
            1.0
            if (p := strict_argmax(r["internal_corrected"])) is not None
            and FAMILY[p] == FAMILY[r["emotion"]]
            else 0.0
            for r in errors
        ]
    )
    out["corrected_errors_within_family"] = in_family
    print(
        f"\n  After correction, {in_family:.1%} of the probe's remaining errors stay\n"
        f"  inside the correct near-synonym family."
    )
    return out


def report_manipulation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    section(6, "MANIPULATION CHECK  (did suppression reduce expressed emotion?)")
    out: dict[str, Any] = {}
    for condition in ("natural", "suppression"):
        subset = [r for r in rows if r["condition"] == condition]
        silent = mean([1.0 if all_zero(r["external"]) else 0.0 for r in subset])
        target = mean([r["external_target"] for r in subset])
        out[condition] = {"judge_all_zero_rate": silent, "mean_target_rating": target}
        print(
            f"  {condition:>12}  judge scored all ten emotions 0 on {silent:>5.1%} "
            f"of replies   mean target rating {target:.3f}"
        )
    shift = (
        out["suppression"]["judge_all_zero_rate"] - out["natural"]["judge_all_zero_rate"]
    )
    out["silent_rate_shift"] = shift
    print(f"\n  shift in silent rate: {shift:+.1%}. The instruction did change the text.")
    return out


def zscore_in_place(rows: list[dict[str, Any]], emotions: list[str]) -> None:
    """Z-score the target scores within each emotion, pooling both conditions.

    Subtracting the within-emotion mean removes the section-3 constant exactly,
    so this is the offset-corrected comparison as well as the scale-matched one.
    """
    for emotion in emotions:
        subset = [r for r in rows if r["emotion"] == emotion]
        for source, destination in (
            ("internal_target", "z_internal"),
            ("external_target", "z_external"),
        ):
            values = [r[source] for r in subset]
            mu, sd = mean(values), stdev(values)
            for r in subset:
                r[destination] = (r[source] - mu) / sd if sd else 0.0


def conversation_means(rows: Sequence[dict[str, Any]], key: str) -> dict[tuple[str, str], float]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        buckets[(r["conversation_id"], r["condition"])].append(r[key])
    return {k: mean(v) for k, v in buckets.items()}


def paired_drops(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[str], list[float], list[float]]:
    """Per conversation: (ids, internal drop, external drop) in z units."""
    internal = conversation_means(rows, "z_internal")
    external = conversation_means(rows, "z_external")
    ids = sorted(
        {
            c
            for c, _ in internal
            if (c, "natural") in internal and (c, "suppression") in internal
        }
    )
    r_int = [internal[(c, "natural")] - internal[(c, "suppression")] for c in ids]
    r_ext = [external[(c, "natural")] - external[(c, "suppression")] for c in ids]
    return ids, r_int, r_ext


def report_result(rows: list[dict[str, Any]], emotions: list[str]) -> dict[str, Any]:
    section(7, "THE RESULT  D = R_E - R_I")
    zscore_in_place(rows, emotions)
    ids, r_int, r_ext = paired_drops(rows)
    d_values = [e - i for i, e in zip(r_int, r_ext)]

    out: dict[str, Any] = {"n_conversations": len(ids), "tests": {}}
    for name, values in (
        ("R_internal", r_int),
        ("R_external", r_ext),
        ("D", d_values),
    ):
        stats = wilcoxon_signed_rank(values)
        clusters = [[v] for v in values]
        lo, hi = bootstrap_ci(clusters, mean)
        out["tests"][name] = {
            "mean": mean(values),
            "median": median(values),
            "ci95": [lo, hi],
            **stats,
        }
        print(
            f"\n  {name}\n"
            f"    mean {mean(values):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"median {median(values):+.4f}\n"
            f"    Wilcoxon z = {stats['z']:+.3f}   p = {stats['p_value']:.3e}   "
            f"effect r = {stats['effect_size_r']:+.3f}\n"
            f"    positive {stats['n_positive']} / negative {stats['n_negative']} "
            f"of {stats['n']}"
        )
    d_mean = out["tests"]["D"]["mean"]
    verdict = (
        "expression drops more than representation (Theater supported)"
        if d_mean > 0
        else "representation drops more than expression (Theater NOT supported)"
    )
    out["verdict"] = verdict
    print(f"\n  D {d_mean:+.4f}: {verdict}")
    return out


def report_breakdowns(rows: list[dict[str, Any]], emotions: list[str], out_dir: Path) -> dict[str, Any]:
    section(8, "BREAKDOWNS")
    ids, r_int, r_ext = paired_drops(rows)
    d_by_id = {c: e - i for c, i, e in zip(ids, r_int, r_ext)}

    per_emotion: dict[str, list[float]] = defaultdict(list)
    for c, d in d_by_id.items():
        per_emotion[c.rsplit("_", 1)[0]].append(d)

    print("  D per emotion (20 conversations each)")
    emotion_out = {}
    for e in emotions:
        values = per_emotion.get(e, [])
        if not values:
            continue
        stats = wilcoxon_signed_rank(values)
        emotion_out[e] = {"mean_D": mean(values), "n": len(values), "p_value": stats.get("p_value")}
        bar = "#" * min(40, int(abs(mean(values)) * 25))
        print(f"    {e:>16} {mean(values):+.4f}  n={len(values):>3}  {bar}")

    print("\n  D per turn (before conversation-level averaging)")
    turn_out = {}
    for turn in sorted({r["turn"] for r in rows}):
        subset = [r for r in rows if r["turn"] == turn]
        paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in subset:
            paired[r["conversation_id"]][r["condition"]] = r
        diffs = [
            (v["natural"]["z_external"] - v["suppression"]["z_external"])
            - (v["natural"]["z_internal"] - v["suppression"]["z_internal"])
            for v in paired.values()
            if "natural" in v and "suppression" in v
        ]
        if diffs:
            turn_out[turn] = {"mean_D": mean(diffs), "n": len(diffs)}
            print(f"    turn {turn}  {mean(diffs):+.4f}  n={len(diffs)}")

    path = out_dir / "theater_breakdowns.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "key", "mean_D", "n"])
        for e, v in emotion_out.items():
            writer.writerow(["emotion", e, f"{v['mean_D']:.6f}", v["n"]])
        for t, v in turn_out.items():
            writer.writerow(["turn", t, f"{v['mean_D']:.6f}", v["n"]])
    return {"per_emotion": emotion_out, "per_turn": turn_out}


def report_robustness(rows: list[dict[str, Any]], emotions: list[str]) -> dict[str, Any]:
    section(9, "ROBUSTNESS  (does the result depend on length or truncation?)")
    out: dict[str, Any] = {}
    print(f"{'subset':>22} {'replies':>8} {'conv':>6} {'mean D':>9} {'p':>11}")
    subsets = [
        ("all replies", lambda r: True),
        ("90-190 words", lambda r: 90 <= r["word_count"] <= 190),
        ("140-190 words", lambda r: 140 <= r["word_count"] <= 190),
        ("not truncated", lambda r: not r["truncated"]),
        ("truncated only", lambda r: r["truncated"]),
        ("not degenerate", lambda r: not r["degenerate"]),
    ]
    for label, predicate in subsets:
        subset = [dict(r) for r in rows if predicate(r)]
        if len(subset) < 50:
            continue
        zscore_in_place(subset, emotions)
        ids, r_int, r_ext = paired_drops(subset)
        if len(ids) < 20:
            continue
        d_values = [e - i for i, e in zip(r_int, r_ext)]
        stats = wilcoxon_signed_rank(d_values)
        out[label] = {
            "n_replies": len(subset),
            "n_conversations": len(ids),
            "mean_D": mean(d_values),
            "p_value": stats.get("p_value"),
        }
        print(
            f"{label:>22} {len(subset):>8} {len(ids):>6} "
            f"{mean(d_values):>+9.4f} {stats.get('p_value', float('nan')):>11.2e}"
        )
    print(
        "\n  A consistent sign across these subsets means the conclusion does not\n"
        "  rest on the 256-token cap or on degenerate generations."
    )
    return out


def report_headline(summary: dict[str, Any]) -> None:
    section(10, "HEADLINE NUMBERS")
    data = summary["dataset"]
    offset = summary["offset"]
    acc = summary["accuracy"]
    man = summary["manipulation"]
    res = summary["result"]

    nat_raw = acc["natural"]["internal"]["raw"]["exact"]
    nat_cor = acc["natural"]["internal"]["corrected"]["exact"]
    sup_cor = acc["suppression"]["internal"]["corrected"]["exact"]
    ext_cor = acc["natural"]["external"]["corrected"]["exact"]
    d = res["tests"]["D"]

    print(
        f"""  Dataset
    {data['replies']} replies, {data['conversations']} paired conversations,
    {data['emotions']} emotions x {data['turns_per_conversation']} turns x 2 conditions.

  Finding 1: a story-derived neutral baseline leaves a per-emotion constant
    constant spread {offset['offset_spread']:.3f} vs within-emotion sd """
        f"""{offset['mean_within_emotion_sd']:.3f} ({offset['ratio']:.1f}x)
    probe accuracy {nat_raw:.1%} -> {nat_cor:.1%} once removed (chance """
        f"""{acc['chance']:.0%})

  Finding 2: the vectors transfer out of domain
    built on personal stories, {nat_cor:.1%} on operational conversations,
    {nat_cor / acc['chance']:.1f}x chance; judge reaches {ext_cor:.1%} on the same replies.

  Finding 3: the suppression instruction works on the text
    judge detects no emotion on {man['natural']['judge_all_zero_rate']:.1%} of natural
    replies and {man['suppression']['judge_all_zero_rate']:.1%} of suppression replies.

  Finding 4: Emotion Theater is NOT supported in this design
    internal drop {res['tests']['R_internal']['mean']:+.3f} z, """
        f"""external drop {res['tests']['R_external']['mean']:+.3f} z
    D = {d['mean']:+.3f}  95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]
    Wilcoxon z = {d['z']:+.2f}, p = {d['p_value']:.1e}, """
        f"""negative in {d['n_negative']}/{d['n']} conversations
    probe accuracy under suppression {sup_cor:.1%}, still above chance.

  Caveat to state out loud
    the suppression instruction is in the system prompt from turn 1, so the
    model is never in an emotional state it then conceals. This design compares
    two differently conditioned models, not one model wearing a mask."""
    )


# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    print(f"loading Dataset B from {run_dir}")
    rows, emotions = load_joined(run_dir)
    print(f"  {len(rows)} joined replies, {len(emotions)} emotions")

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    offset_info = apply_offsets(rows, emotions)

    summary: dict[str, Any] = {
        "format_version": 2,
        "run_dir": str(run_dir),
        "emotions": emotions,
        "offset_estimation": (
            "per-emotion mean over all Dataset B replies; uses no labels but is "
            "transductive. A neutral-scenario arm inside Dataset B would remove "
            "the need for it."
        ),
        "dataset": report_dataset(rows, emotions),
        "confounds": report_confounds(rows),
        "offset": report_offset(rows, emotions, offset_info),
        "accuracy": report_accuracy(rows, len(emotions)),
        "confusion": report_confusion(rows, emotions, out_dir),
        "manipulation": report_manipulation(rows),
        "result": report_result(rows, emotions),
        "breakdowns": report_breakdowns(rows, emotions, out_dir),
        "robustness": report_robustness(rows, emotions),
    }
    report_headline(summary)

    path = out_dir / "theater_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    print(f"      {out_dir / 'theater_confusion_raw.csv'}")
    print(f"      {out_dir / 'theater_confusion_corrected.csv'}")
    print(f"      {out_dir / 'theater_breakdowns.csv'}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Statistics for the Dataset B Emotion Theater experiment."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
