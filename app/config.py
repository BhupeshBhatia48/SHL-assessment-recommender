"""
Central configuration. All tunables live here so behavior can be adjusted
without hunting through the pipeline modules.

LLM provider is selected via the LLM_PROVIDER env var so the same codebase
works with either free-tier option mentioned in the assignment without code
changes -- just flip the env var and provide the matching key.
"""
import os

# --- LLM provider ---------------------------------------------------------
# "gemini" or "groq". Both are free-tier and fast enough to comfortably fit
# two calls inside the 30-second per-request budget.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))

# --- Retrieval -------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "18"))
MIN_RECOMMENDATIONS = 1
MAX_RECOMMENDATIONS = 10
DEFAULT_RECOMMENDATIONS_ON_FALLBACK = 3

# --- Conversation / turn budget --------------------------------------------
# The assignment states "the evaluator caps each conversation at 8 turns
# including user & assistant". Taken literally as "8 total messages" this
# would mean at most 4 user + 4 assistant messages -- but the longest
# provided sample trace (C9) has 7 user/assistant exchanges (14 total
# messages), and C3 has 10. Both would violate a strict 8-message cap. Since
# these are SHL's own ground-truth examples, we trust that evidence over the
# ambiguous wording and interpret "8 turns" as 8 user-assistant EXCHANGES
# (i.e. up to 16 total messages), which comfortably fits every provided
# trace with margin. MAX_TOTAL_MESSAGES below is expressed in raw message
# count to match how we actually measure `len(messages)` from the request.
MAX_TOTAL_MESSAGES = 16

# How many of the most recent messages to actually include when building the
# LLM prompt context. Keeps prompt size (and therefore latency) bounded even
# though the full history is always received.
MAX_HISTORY_MESSAGES_FOR_PROMPT = 16

# --- Guardrails --------------------------------------------------------------
# Minimum constraint signal required before the router will allow a
# "recommend" intent through, regardless of what the LLM analyzer claims.
# This is a code-level floor, not just a prompt instruction -- see
# app/pipeline/router.py.
REQUIRE_ROLE_OR_SKILL_BEFORE_RECOMMEND = True
