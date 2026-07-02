"""
LLMs are asked to return JSON-only, but "asked to" isn't "guaranteed to" --
models sometimes wrap output in ```json fences, add a stray sentence before
or after, or use smart quotes. This module extracts the JSON object
defensively so a formatting slip doesn't crash the pipeline.
"""
import json
import re


class JSONExtractionError(Exception):
    pass


def extract_json(text: str) -> dict:
    text = text.strip()

    # Strip markdown code fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the first {...} block via brace matching.
    start = text.find("{")
    if start == -1:
        raise JSONExtractionError(f"No JSON object found in LLM output: {text[:200]!r}")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise JSONExtractionError(
                        f"Found JSON-like block but it didn't parse: {e}"
                    ) from e

    raise JSONExtractionError(f"Unbalanced braces in LLM output: {text[:200]!r}")
