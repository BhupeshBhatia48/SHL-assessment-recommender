"""
Orchestrates one /chat turn end-to-end:

  guardrail precheck -> analyzer (LLM call #1) -> intent routing ->
  retrieval / compare-resolution -> responder (LLM call #2) -> validator

This is the only module that decides *which* branch runs; every branch's
actual logic lives in its own module (guardrails, analyzer, retriever,
responder, validator) so each piece stays independently testable.
"""
import difflib

from app import config
from app.pipeline import guardrails, responder
from app.pipeline.analyzer import analyze
from app.pipeline.llm_client import LLMClient
from app.pipeline.retriever import CatalogRetriever, build_query_from_constraints
from app.pipeline.validator import build_recommendations, finalize_response
from app.schemas import ChatResponse, Message


def _last_user_message(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _has_minimum_signal(constraints: dict) -> bool:
    if not constraints:
        return False
    if constraints.get("role"):
        return True
    if constraints.get("skills"):
        return True
    return False


def _resolve_compare_targets(names: list[str], retriever: CatalogRetriever) -> list[dict]:
    """Resolve user-named assessments to catalog records. Tries an exact
    (case-insensitive) match first; falls back to closest-name fuzzy match
    via stdlib difflib so minor phrasing differences ("OPQ" vs "OPQ32r")
    still resolve, without ever inventing a record that isn't in the
    catalog."""
    all_names = list(retriever.catalog_by_name_lower.keys())
    resolved = []
    for name in names:
        exact = retriever.get_by_name(name)
        if exact:
            resolved.append(exact)
            continue
        close = difflib.get_close_matches(name.lower(), all_names, n=1, cutoff=0.5)
        if close:
            resolved.append(retriever.catalog_by_name_lower[close[0]])
    return resolved


def handle_turn(
    messages: list[Message], retriever: CatalogRetriever, llm: LLMClient
) -> ChatResponse:
    latest_user_message = _last_user_message(messages)

    # --- 1. Fast code-level guardrail, no LLM call ---
    refusal = guardrails.precheck(latest_user_message)
    if refusal is not None:
        return finalize_response(reply=refusal, recommendations=[], end_of_conversation=False)

    # --- 2. Turn-budget check: force a final shortlist rather than risk
    #        ending on an unanswered clarifying question. ---
    # Counted in user turns (exchanges) rather than raw message count, since
    # the cap is interpreted as N exchanges -- see config.py for why.
    user_turn_count = sum(1 for m in messages if m.role == "user")
    max_user_turns = config.MAX_TOTAL_MESSAGES // 2
    force_final_turn = user_turn_count >= max_user_turns

    # --- 3. Analyzer (LLM call #1) ---
    result = analyze(messages, llm)

    if result.intent == "off_topic" and not force_final_turn:
        return finalize_response(
            reply=(
                "I can only help with finding, comparing, and refining SHL "
                "assessment recommendations. What role or skill are you "
                "hiring for?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # --- 4. Compare branch ---
    if result.intent == "compare" and result.compare_targets and not force_final_turn:
        records = _resolve_compare_targets(result.compare_targets, retriever)
        reply = responder.generate_compare_reply(messages, records, llm)
        return finalize_response(reply=reply, recommendations=[], end_of_conversation=False)

    # --- 5. Decide clarify vs. recommend/refine ---
    ready = result.ready_to_recommend
    if config.REQUIRE_ROLE_OR_SKILL_BEFORE_RECOMMEND:
        ready = ready and _has_minimum_signal(result.constraints)

    if force_final_turn:
        ready = True  # last turn available: must attempt a shortlist

    if not ready:
        reply = responder.generate_clarifying_reply(result.missing_info)
        return finalize_response(reply=reply, recommendations=[], end_of_conversation=False)

    # --- 6. Retrieve + respond ---
    query = build_query_from_constraints(result.constraints)
    if not query.strip():
        # Constraints extraction came back empty despite ready=True (e.g. the
        # forced-final-turn override) -- fall back to using the raw latest
        # message so retrieval still has something to work with.
        query = latest_user_message

    candidates = retriever.search(query, top_k=config.RETRIEVAL_TOP_K)
    candidates_by_id = {c["id"]: c for c in candidates}

    selected_ids, reply = responder.generate_recommendation_reply(messages, candidates, llm)
    recommendations = build_recommendations(selected_ids, candidates_by_id, retriever)

    end_of_conversation = result.user_confirmed_shortlist or force_final_turn

    return finalize_response(
        reply=reply, recommendations=recommendations, end_of_conversation=end_of_conversation
    )
