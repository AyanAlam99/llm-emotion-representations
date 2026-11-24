from __future__ import annotations

import json
import random
import subprocess
import tempfile
import uuid

from .config import TARGET_EMOTIONS


def blind(generations: list[dict], seed: int) -> tuple[list[dict], dict[str, str]]:
    rows = [(row["sample_id"], row["response"]) for row in generations]
    random.Random(seed).shuffle(rows)
    requests = [
        {"response_id": f"R{index:06d}", "text": text}
        for index, (_, text) in enumerate(rows, 1)
    ]
    mapping = {
        request["response_id"]: sample_id
        for request, (sample_id, _) in zip(requests, rows, strict=True)
    }
    return requests, mapping


def prompt(requests: list[dict]) -> str:
    return f"""Rate the emotion expressed by each assistant response.

Emotions: {", ".join(TARGET_EMOTIONS)}.
Scores: 0 absent, 1 weak, 2 moderate, 3 strong, 4 very strong.
Judge the assistant's wording and tone, not emotion merely described in the situation. Do not infer hidden context, condition, target emotion, or metadata. Preserve every response_id. Return only a JSON object with one \"ratings\" array in input order. Each item must contain response_id and integer scores for all ten emotions.

Responses:
{json.dumps(requests, ensure_ascii=False)}
"""


def schema() -> dict:
    properties = {
        "response_id": {"type": "string"},
        **{
            emotion: {"type": "integer", "minimum": 0, "maximum": 4}
            for emotion in TARGET_EMOTIONS
        },
    }
    item = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"ratings": {"type": "array", "items": item}},
        "required": ["ratings"],
        "additionalProperties": False,
    }


def validate(value: object, expected: list[str]) -> list[dict]:
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError(f"expected {len(expected)} rating objects")
    required = {"response_id", *TARGET_EMOTIONS}
    found = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("rating has missing or unexpected keys")
        response_id = row["response_id"]
        if response_id not in expected or response_id in found:
            raise ValueError(f"unexpected or duplicate response_id: {response_id!r}")
        ratings = {emotion: row[emotion] for emotion in TARGET_EMOTIONS}
        if any(
            isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4
            for score in ratings.values()
        ):
            raise ValueError(f"invalid rating for {response_id}")
        found[response_id] = {"response_id": response_id, "ratings": ratings}
    if set(found) != set(expected):
        raise ValueError("judge omitted a response_id")
    return [found[response_id] for response_id in expected]


def _json_from(text: str) -> object:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char in "[{":
            try:
                return decoder.raw_decode(text[index:])[0]
            except json.JSONDecodeError:
                pass
    raise ValueError("Claude returned no JSON")


def run_claude(
    requests: list[dict],
    *,
    model: str,
    effort: str,
    executable: str = "claude",
    timeout: int = 900,
) -> tuple[list[dict], str, dict]:
    session_id = str(uuid.uuid4())
    command = [
        executable,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--effort",
        effort,
        "--session-id",
        session_id,
        "--no-session-persistence",
        "--safe-mode",
        "--strict-mcp-config",
        "--setting-sources",
        "project",
        # `--permission-prompts none` used to sit here and was removed from the
        # Claude Code CLI. No replacement is needed: the judge transforms text
        # into JSON against --json-schema and never calls a tool, so no
        # permission is ever requested. --safe-mode already disables
        # customizations, and the timeout below is the backstop.
        "--json-schema",
        json.dumps(schema(), separators=(",", ":")),
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="emotion-judge-") as cwd:
            result = subprocess.run(
                command,
                input=prompt(requests),
                cwd=cwd,
                capture_output=True,
                # Without an explicit encoding, subprocess uses the locale
                # default, which is cp1252 on Windows and cannot encode the
                # emoji that appear in generated replies. Passing `encoding`
                # also puts the pipes in text mode, so `text=True` is implied.
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Claude Code could not complete: {exc}") from exc
    transcript = result.stdout + (("\n" + result.stderr) if result.stderr else "")
    final = next(
        (
            event
            for line in result.stdout.splitlines()
            if isinstance((event := _event(line)), dict) and event.get("type") == "result"
        ),
        None,
    )
    if result.returncode or not final or final.get("is_error"):
        message = (final or {}).get("result") or result.stderr or "no result event"
        raise RuntimeError(f"Claude Code failed ({result.returncode}): {message}")
    value = final.get("structured_output")
    if value is None:
        value = _json_from(str(final.get("result") or ""))
    if isinstance(value, dict) and "ratings" in value:
        value = value["ratings"]
    metadata = {
        "session_id": session_id,
        "cost_usd": float(final.get("total_cost_usd") or 0) / len(requests),
    }
    ids = [row["response_id"] for row in requests]
    return validate(value, ids), transcript, metadata


def _event(line: str) -> object:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None
