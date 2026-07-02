"""
Analyzer stage: turns the full conversation history into structured intent +
accumulated constraints. This is LLM call #1 of the (at most) 2 calls per
turn budget.

Re-deriving the FULL constraint set from the WHOLE history on every call
(rather than diffing just the newest message) is what makes "refine" work
without any special-case code path or server-side conversation memory: a
later message that adds or changes a constraint naturally merges into this
same extraction step every time, since the API is stateless and the entire
history is reprocessed each call anyway.
"""
from dataclasses import dataclass, field
from pathlib import Path

from app.pipeline.json_utils import extract_json, JSONExtractionError
from app.pipeline.llm_client import LLMClient, LLMError
from app.schemas import Message

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyzer_prompt.txt"
ANALYZER_SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class AnalyzerResult:
    intent: str  # clarify | recommend | refine | compare | off_topic
    constraints: dict = field(default_factory=dict)
    compare_targets: list = field(default_factory=list)
    missing_info: list = field(default_factory=list)
    ready_to_recommend: bool = False
    user_confirmed_shortlist: bool = False


def _format_history(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        speaker = "User" if m.role == "user" else "Agent"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


def _fallback_result(messages: list[Message]) -> AnalyzerResult:
    """Used if the LLM call or JSON parsing fails outright. Falls back to a
    conservative "clarify" so the pipeline never crashes -- a slightly less
    helpful reply beats a 500 error against the 30s-timeout evaluator."""
    return AnalyzerResult(
        intent="clarify",
        constraints={},
        missing_info=["role or skills you're hiring for"],
        ready_to_recommend=False,
    )


def analyze(messages: list[Message], llm: LLMClient) -> AnalyzerResult:
    history_text = _format_history(messages)
    user_prompt = (
        "Conversation so far:\n\n"
        f"{history_text}\n\n"
        "Output the JSON object described in your instructions now."
    )

    try:
        raw = llm.complete(ANALYZER_SYSTEM_PROMPT, user_prompt)
        data = extract_json(raw)
    except (LLMError, JSONExtractionError):
        return _fallback_result(messages)

    return AnalyzerResult(
        intent=data.get("intent", "clarify"),
        constraints=data.get("constraints", {}) or {},
        compare_targets=data.get("compare_targets", []) or [],
        missing_info=data.get("missing_info", []) or [],
        ready_to_recommend=bool(data.get("ready_to_recommend", False)),
        user_confirmed_shortlist=bool(data.get("user_confirmed_shortlist", False)),
    )
