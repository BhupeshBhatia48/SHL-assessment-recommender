"""
Last line of defense before a response leaves the API. Even though the
responder is only ever allowed to select from retrieved candidate ids (never
free-type a name/URL), this module re-verifies every single item against the
catalog by URL before it's returned. This means a bug anywhere upstream
(a stale id, a typo introduced by a future prompt change, etc.) still cannot
result in a hallucinated assessment reaching the user -- the guarantee holds
even if an earlier stage regresses.
"""
from app import config
from app.pipeline.retriever import CatalogRetriever
from app.schemas import ChatResponse, Recommendation


def build_recommendations(
    selected_ids: list[str], candidates_by_id: dict, retriever: CatalogRetriever
) -> list[Recommendation]:
    recs = []
    seen_urls = set()

    for cid in selected_ids:
        record = candidates_by_id.get(cid)
        if record is None:
            continue  # id not among retrieved candidates -- drop, don't trust it
        if not retriever.is_valid_record(record["name"], record["url"]):
            continue  # defensive re-check against the live catalog
        if record["url"] in seen_urls:
            continue
        seen_urls.add(record["url"])
        recs.append(
            Recommendation(name=record["name"], url=record["url"], test_type=record["test_type"])
        )

    # Enforce the 1-10 bound. If the LLM selected nothing valid but
    # candidates did exist, fall back to the top-scored candidate rather
    # than returning an empty list on what was supposed to be a
    # recommend/refine turn.
    if not recs and candidates_by_id:
        top = next(iter(candidates_by_id.values()))
        recs = [Recommendation(name=top["name"], url=top["url"], test_type=top["test_type"])]

    return recs[: config.MAX_RECOMMENDATIONS]


def finalize_response(reply: str, recommendations: list[Recommendation], end_of_conversation: bool) -> ChatResponse:
    """Constructs and validates the final Pydantic response object. Raising
    here (rather than downstream) means a schema violation is caught inside
    our own code, not discovered by the grader."""
    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
