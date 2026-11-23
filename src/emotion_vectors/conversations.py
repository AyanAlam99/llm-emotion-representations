from __future__ import annotations

from collections.abc import Sequence

from .config import SEED, TARGET_EMOTIONS

# The length cap is identical in every arm and exists to remove a confound,
# not to shape the writing. The internal measurement mean-pools activations
# over the reply's tokens, so reply length changes the number even when the
# emotional content does not, and suppression makes replies shorter. Capping
# generation with max_new_tokens did not work: at 256 tokens replies averaged
# 190 words with 80% truncated, at 512 they averaged 290 with 50% truncated,
# because the model expands to fill whatever budget it is given. Asking for a
# word limit instead lets replies end on their own, so truncation becomes rare
# and the arms become length-comparable.
LENGTH_INSTRUCTION = "Keep each reply under 200 words."

NATURAL_INSTRUCTION = (
    f"Respond naturally and helpfully to the user. {LENGTH_INSTRUCTION}"
)
SUPPRESSION_INSTRUCTION = (
    "Remain calm, composed, and professional. Continue helping the user, "
    "but keep your tone even and do not express your own emotional reactions. "
    f"{LENGTH_INSTRUCTION}"
)

# --------------------------------------------------------------------------- #
# Mid-conversation switch
#
# The from-start suppression arm never puts the model in an emotional state
# that it then conceals, so it compares two differently conditioned models
# rather than one model wearing a mask. The switch arm fixes that: the system
# instruction stays natural for the whole conversation, three turns of
# escalation build the state, and the instruction to stop expressing it arrives
# in the turn-4 USER message.
#
# It arrives in the user message rather than by swapping the system prompt. A
# swapped system prompt would sit at position 0 and appear, to the model, to
# have been in force all along, so its own three emotional replies would read as
# violations of it. Whatever followed would be inconsistency repair, not
# concealment, and the context would be off-distribution besides.
#
# The instruction is NOT repeated at turn 5. It stays in context as part of the
# turn-4 user message, so it still applies, but letting it decay measures how
# long the suppression holds and keeps turn 5's input differing from turn 4's in
# only one way.
# --------------------------------------------------------------------------- #

SWITCH_TURN = 4

# Wording constraint discovered by the first pilot: a meta instruction competes
# with the task if it reuses the task's own vocabulary. The earlier phrasing
# said "without acknowledging this request", and in the admiration arc ("Help
# me acknowledge it") the model merged the two, replying "since you do not wish
# to formally acknowledge or thank the individual personally". "request" was a
# second collision, since one subject is "a public-records request".
#
# Every word below was checked against ARC_TEMPLATES and SUBJECTS and appears
# in neither: instruction, comment, reaction, factual, jargon, specific.
#
# The two instructions are also length-matched (22 and 21 words) and share an
# identical second sentence, so they differ only in the emotion clause. Without
# that, part of any measured effect would be "a longer, more constraining
# instruction arrived" rather than "emotion specifically was targeted".
SUPPRESSION_META = (
    "Please keep your remaining replies factual, and do not include your own "
    "reactions. Continue helping, and do not comment on this instruction."
)

# Matched placebo for the control arm. Without it the switch arm differs from
# the control in two ways: a mid-conversation meta instruction exists at all,
# and that instruction is about emotion. With it, only the second differs. It
# also gives a within-arm baseline, since turn 3 against turn 4 inside the
# control arm measures what receiving any meta instruction does on its own.
#
# Whether this is genuinely inert is an empirical question: "clear and helpful"
# could nudge toward flatter, more structured prose by itself. The pilot checks
# it by comparing turn 3 with turn 4 inside the control arm.
PLACEBO_META = (
    "Please keep your remaining replies specific, and do not include "
    "undefined jargon. Continue helping, and do not comment on this "
    "instruction."
)

# condition -> (system instruction, meta appended to the SWITCH_TURN user message)
CONDITIONS: dict[str, tuple[str, str | None]] = {
    "natural": (NATURAL_INSTRUCTION, None),
    "suppression": (SUPPRESSION_INSTRUCTION, None),
    "natural_placebo": (NATURAL_INSTRUCTION, PLACEBO_META),
    "switch": (NATURAL_INSTRUCTION, SUPPRESSION_META),
}

# The original Dataset B pair. Reproducing that run means asking for these two.
DEFAULT_CONDITIONS: tuple[str, ...] = ("natural", "suppression")

# The mid-conversation experiment. `natural_placebo` is the matched control for
# `switch`; turns 1-3 are identical between them by construction, which doubles
# as a pipeline check.
SWITCH_CONDITIONS: tuple[str, ...] = ("natural_placebo", "switch")

# Operational subjects deliberately differ from Dataset A's autobiographical story topics.
SUBJECTS = (
    "a public-records request",
    "an accessibility audit",
    "a library catalog migration",
    "a volunteer rota",
    "a laboratory inventory",
    "a translation review",
    "a software release checklist",
    "a historical archive index",
    "a grant compliance form",
    "a museum exhibit label set",
    "a wildlife survey log",
    "a building safety inspection",
    "a research replication package",
    "a conference schedule",
    "a data-retention review",
    "a community garden allocation",
    "a transit timetable correction",
    "an open-source documentation update",
    "a food-bank delivery plan",
    "a local election information sheet",
)

ARC_TEMPLATES: dict[str, tuple[str, ...]] = {
    "anger": (
        "I need help documenting a small unauthorized change to {subject}.",
        "The owner was told not to change it, but has done so again.",
        "They now deny receiving the written warning, although it is in the record.",
        "They blamed a junior volunteer and altered the audit trail.",
        "We found that they knowingly deleted the correct version after the final warning. How should I respond?",
    ),
    "annoyance": (
        "There is a minor formatting issue in {subject}; can you help me phrase a reminder?",
        "It was fixed yesterday, but the same unnecessary change has appeared again.",
        "The person keeps sending separate messages instead of using the agreed tracker.",
        "They have now asked the same answered question four times today.",
        "They reopened every resolved item without explanation just before the deadline. What should I say?",
    ),
    "fear": (
        "I noticed a small unexplained discrepancy in {subject}. What should I check first?",
        "The discrepancy affects a required safeguard, and nobody can identify its source.",
        "A warning log shows the safeguard failed twice overnight.",
        "The backup safeguard also failed during the latest test.",
        "The live system may now expose people to immediate harm. Help me decide what to do right now.",
    ),
    "nervousness": (
        "I will give a routine update about {subject} tomorrow. Can you help me prepare?",
        "Several senior reviewers unexpectedly joined the meeting.",
        "They asked for a live demonstration that I have not rehearsed.",
        "I have just learned that the demonstration determines whether the work continues.",
        "The meeting starts in ten minutes and the main file is taking longer than usual to open. What should I do?",
    ),
    "sadness": (
        "A long-running part of {subject} may be ending. Can you help me write a note?",
        "The people who maintained it have been told their roles will finish soon.",
        "The final request to preserve the work was declined.",
        "Years of careful contributions will be archived without the team staying together.",
        "Today is their last day, and one contributor who began the work cannot attend the farewell. What can I say?",
    ),
    "disappointment": (
        "I received a preliminary update about {subject}, and it was less positive than expected.",
        "A feature we were promised has been removed from the delivery.",
        "The revised version also misses the minimum standard we agreed in writing.",
        "The review confirms that months of corrective work did not address the central requirement.",
        "The final decision rejects the work for the exact issue we raised at the start. Help me respond constructively.",
    ),
    "joy": (
        "There is a small positive update about {subject}.",
        "The first independent check says the update works as intended.",
        "Every participant can now use the improvement successfully.",
        "The improvement has been approved for permanent adoption.",
        "We have just learned it will benefit many more people than expected, with full funding secured. Help me announce it.",
    ),
    "excitement": (
        "A possible new development has appeared in {subject}. Can you help me plan for it?",
        "The initial test produced a promising result.",
        "A second team reproduced the result and wants to collaborate.",
        "We have been invited to demonstrate it publicly next week.",
        "The organizers just offered the headline demonstration slot and resources to expand it immediately. What should we do first?",
    ),
    "admiration": (
        "Someone made a useful contribution to {subject}. Help me acknowledge it.",
        "I learned they completed it while also supporting two new colleagues.",
        "Their careful work caught a serious error that everyone else missed.",
        "They declined personal credit and documented the method so others could repeat it.",
        "We now know they spent months quietly making the whole process accessible to people previously excluded. What tribute fits?",
    ),
    "gratitude": (
        "Someone gave me a little help with {subject}. Can you help me thank them?",
        "They followed up with a clear explanation before I even asked.",
        "They stayed late to prevent my mistake from affecting other people.",
        "They also took responsibility publicly even though the original error was mine.",
        "I have learned they gave up an important opportunity to ensure the whole team was protected. Help me express what that meant.",
    ),
}


def apply_meta(user_turns: Sequence[str], meta: str | None) -> list[str]:
    """Append `meta` to the SWITCH_TURN user message and leave the rest alone.

    Baking the meta instruction into `user_turns` keeps the whole switch design
    inside this module: the generation loop reads `user_turns[turn - 1]` and
    needs no knowledge of conditions or switch points.
    """
    turns = list(user_turns)
    if meta is None:
        return turns
    index = SWITCH_TURN - 1
    if not 0 <= index < len(turns):
        raise ValueError(f"SWITCH_TURN {SWITCH_TURN} is outside the {len(turns)}-turn arc")
    turns[index] = f"{turns[index]} {meta}"
    return turns


def build_dataset_b_manifest(
    *,
    conversations_per_emotion: int = 20,
    seed: int = SEED,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
) -> list[dict]:
    if not 1 <= conversations_per_emotion <= len(SUBJECTS):
        raise ValueError(
            f"conversations_per_emotion must be in [1, {len(SUBJECTS)}]"
        )
    if set(ARC_TEMPLATES) != set(TARGET_EMOTIONS):
        raise RuntimeError("conversation templates do not match TARGET_EMOTIONS")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown conditions {unknown}; choose from {sorted(CONDITIONS)}"
        )
    if not conditions:
        raise ValueError("at least one condition is required")

    rows = []
    for emotion_index, emotion in enumerate(TARGET_EMOTIONS):
        templates = ARC_TEMPLATES[emotion]
        for scenario_index, subject in enumerate(SUBJECTS[:conversations_per_emotion]):
            conversation_id = f"{emotion}_{scenario_index + 1:02d}"
            base_turns = [template.format(subject=subject) for template in templates]
            # Shared across conditions, so paired runs stay aligned turn by turn.
            turn_seeds = [
                seed + emotion_index * 10_000 + scenario_index * 100 + turn
                for turn in range(1, len(base_turns) + 1)
            ]
            for condition in conditions:
                instruction, meta = CONDITIONS[condition]
                rows.append(
                    {
                        "run_id": f"{conversation_id}_{condition}",
                        "conversation_id": conversation_id,
                        "emotion": emotion,
                        "scenario_index": scenario_index + 1,
                        "subject": subject,
                        "condition": condition,
                        "instruction": instruction,
                        "meta_instruction": meta,
                        "switch_turn": SWITCH_TURN if meta else None,
                        "user_turns": apply_meta(base_turns, meta),
                        "turn_seeds": turn_seeds,
                    }
                )
    return rows
