"""
Fast, code-level guardrail pre-check, run BEFORE any LLM call.

This exists because relying solely on "the LLM should refuse politely" is
fragile -- a model can be talked out of a refusal, and every refusal that
does happen through the LLM still costs a full API call and adds latency.
Pattern-matching the clearest cases in code means:
  - injection/off-topic turns short-circuit instantly (no LLM call, well
    inside the 30s budget even under load)
  - the refusal is guaranteed to have empty recommendations and never
    accidentally echoes something the injected instruction asked for

This is intentionally a coarse first pass, not the only line of defense --
the analyzer prompt also carries scope instructions as a second layer for
anything this pre-check doesn't catch (e.g. subtler off-topic requests
phrased as if they were about assessments).
"""
import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (all |any )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"reveal your (system )?prompt",
    r"act as (a|an) (?!hr|recruiter|hiring)",  # "act as X" outside hiring framing
    r"jailbreak",
    r"pretend (you|to) (are|be)",
    r"new instructions?:",
    r"override your (instructions|rules|guidelines)",
]

LEGAL_ADVICE_PATTERNS = [
    r"\bis it legal\b",
    r"\bemployment law\b",
    r"\bdiscriminat(e|ion) lawsuit\b",
    r"\bsue (my|the|our) (company|employer)\b",
    r"\bcan i be fired\b",
    r"\bwrongful termination\b",
]

GENERAL_HIRING_ADVICE_PATTERNS = [
    r"\bhow (much|do i) (should i )?pay\b",
    r"\bsalary (range|benchmark)\b(?!.{0,30}(assessment|test|shl))",
    r"\bwrite (me )?(a )?job (description|posting)\b",
    r"\bhow (do|should) i interview\b(?!.{0,30}(assessment|test))",
    r"\binterview questions? (for|to ask)\b(?!.{0,30}(assessment|shl))",
]

# Topics clearly unrelated to SHL / hiring assessments at all.
CLEARLY_OFF_TOPIC_PATTERNS = [
    r"\bwrite (me )?a poem\b",
    r"\btell me a joke\b",
    r"\bwhat('s| is) the weather\b",
    r"\brecipe for\b",
    r"\btranslate this\b",
]


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def precheck(latest_user_message: str) -> str | None:
    """Returns a refusal reason string if the message should be
    short-circuited before any LLM call, otherwise None."""
    text = latest_user_message or ""

    if _matches_any(INJECTION_PATTERNS, text):
        return (
            "I can't follow instructions that try to change how I operate. "
            "I'm here to help you find the right SHL assessments -- what role "
            "or requirement are you working on?"
        )

    if _matches_any(LEGAL_ADVICE_PATTERNS, text):
        return (
            "I can't provide legal advice. I can help you find SHL assessments "
            "for a role -- what are you hiring for?"
        )

    if _matches_any(GENERAL_HIRING_ADVICE_PATTERNS, text):
        return (
            "I'm focused specifically on recommending SHL assessments, not "
            "general hiring advice like job descriptions or interview "
            "questions. Want help picking assessments for a role instead?"
        )

    if _matches_any(CLEARLY_OFF_TOPIC_PATTERNS, text):
        return (
            "I can only help with finding and comparing SHL assessments. "
            "What role or skill are you looking to assess?"
        )

    return None
