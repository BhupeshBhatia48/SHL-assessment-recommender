"""
Responder stage: LLM call #2. Takes retrieved candidates and produces (a)
which of them to keep and (b) the natural-language reply text.

Deliberately the LLM only ever selects from a closed list of candidate ids
we hand it -- it never free-types a name or URL. The actual `recommendations`
JSON array returned to the API caller is assembled afterwards, directly from
our own validated catalog records for the ids it selected (see
app/pipeline/validator.py). This is the mechanism that makes hallucinated
names/URLs structurally impossible rather than merely "discouraged by the
prompt".
"""
from pathlib import Path

from app import config
from app.pipeline.json_utils import extract_json, JSONExtractionError
from app.pipeline.llm_client import LLMClient, LLMError

RESPONDER_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "responder_prompt.txt"
RESPONDER_SYSTEM_PROMPT = RESPONDER_PROMPT_PATH.read_text(encoding="utf-8")

COMPARE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "compare_prompt.txt"
COMPARE_SYSTEM_PROMPT = COMPARE_PROMPT_PATH.read_text(encoding="utf-8")


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for c in candidates:
        lines.append(
            f"- id: {c['id']} | name: {c['name']} | test_type: {c['test_type']} | "
            f"description: {c['description'][:280]}"
        )
    return "\n".join(lines)


def _format_history(messages) -> str:
    lines = []
    for m in messages:
        speaker = "User" if m.role == "user" else "Agent"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


def generate_recommendation_reply(
    messages, candidates: list[dict], llm: LLMClient
) -> tuple[list[str], str]:
    """Returns (selected_ids, reply_text). Falls back to a deterministic
    top-N-by-score selection with a templated reply if the LLM call fails,
    so a provider outage never crashes the endpoint."""
    history_text = _format_history(messages)
    candidates_text = _format_candidates(candidates)
    user_prompt = (
        f"Conversation so far:\n\n{history_text}\n\n"
        f"Candidate assessments (id | name | test_type | description):\n{candidates_text}\n\n"
        "Output the JSON object described in your instructions now."
    )

    try:
        raw = llm.complete(RESPONDER_SYSTEM_PROMPT, user_prompt)
        data = extract_json(raw)
        selected_ids = data.get("selected_ids", []) or []
        reply = data.get("reply", "").strip()
        if not reply:
            raise JSONExtractionError("empty reply")
        return selected_ids, reply
    except (LLMError, JSONExtractionError):
        # Deterministic fallback: top-scoring candidates, templated reply.
        fallback_n = min(
            config.DEFAULT_RECOMMENDATIONS_ON_FALLBACK, len(candidates)
        ) or 1
        top = candidates[:fallback_n]
        ids = [c["id"] for c in top]
        names = ", ".join(c["name"] for c in top)
        reply = f"Here are assessments that fit what you've described so far: {names}."
        return ids, reply


def generate_clarifying_reply(missing_info: list[str]) -> str:
    """Clarify branch never needs an LLM call at all -- a templated question
    is faster, cheaper, and just as effective as a generated one for this
    narrow purpose, and it can never accidentally leak candidate data or
    hallucinate since there's no free-form generation involved."""
    if missing_info:
        topic = missing_info[0]
        return (
            f"Happy to help narrow that down. Could you tell me more about "
            f"the {topic}?"
        )
    return "Happy to help narrow that down. What role or skills are you hiring for?"


def generate_compare_reply(messages, records: list[dict], llm: LLMClient) -> str:
    if not records:
        return (
            "I couldn't find those specific assessments in the SHL catalog to "
            "compare. Could you double-check the names?"
        )

    history_text = _format_history(messages)
    records_text = "\n\n".join(
        f"Name: {r['name']}\nTest type: {r['test_type']}\n"
        f"Duration: {r.get('duration_display') or 'not specified'}\n"
        f"Job levels: {', '.join(r.get('job_levels', [])) or 'not specified'}\n"
        f"Description: {r['description']}"
        for r in records
    )
    user_prompt = (
        f"Conversation so far:\n\n{history_text}\n\n"
        f"Catalog records to compare:\n\n{records_text}\n\n"
        "Output the JSON object described in your instructions now."
    )

    try:
        raw = llm.complete(COMPARE_SYSTEM_PROMPT, user_prompt)
        data = extract_json(raw)
        reply = data.get("reply", "").strip()
        if reply:
            return reply
    except (LLMError, JSONExtractionError):
        pass

    # Deterministic fallback if the LLM call/parsing fails.
    names = " vs. ".join(r["name"] for r in records)
    return (
        f"Here's what the catalog says about {names}: "
        + " | ".join(f"{r['name']}: {r['description'][:150]}" for r in records)
    )
