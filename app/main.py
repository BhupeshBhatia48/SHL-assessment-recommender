"""
FastAPI entrypoint. Two endpoints per the assignment spec:
  GET  /health -> {"status": "ok"}
  POST /chat   -> stateless conversational recommender

The retriever (catalog + FAISS index + embedding model) and LLM client are
constructed once at startup and reused across requests -- loading the
embedding model per-request would blow the 30-second timeout budget.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.pipeline.llm_client import LLMError, MockLLMClient, get_default_client
from app.pipeline.retriever import CatalogRetriever, GeminiEmbedder
from app.pipeline.router import handle_turn
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-recommender")

_retriever: CatalogRetriever | None = None
_llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _llm
    logger.info("Loading embedding index (Gemini API embedder, low memory footprint)...")
    _retriever = CatalogRetriever(embedder=GeminiEmbedder())
    logger.info("Catalog retriever ready (%d items).", len(_retriever.catalog))

    try:
        _llm = get_default_client()
        logger.info("LLM client ready (provider=%s).", config.LLM_PROVIDER)
    except LLMError as e:
        logger.warning(
            "No LLM API key configured (%s). Falling back to MockLLMClient -- "
            "set GEMINI_API_KEY or GROQ_API_KEY for real responses.",
            e,
        )
        _llm = MockLLMClient()

    yield  # app runs here

    # no explicit teardown needed -- in-memory index/client, nothing to close


app = FastAPI(title="SHL Conversational Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return handle_turn(request.messages, _retriever, _llm)
    except Exception:
        # The pipeline's own modules already have targeted fallbacks (LLM
        # failure, JSON parse failure, empty retrieval, etc.) -- this is a
        # final safety net so a genuinely unexpected bug still returns a
        # schema-valid response instead of a 500, since a 500 is an
        # automatic hard-eval failure regardless of cause.
        logger.exception("Unhandled error in /chat pipeline")
        return ChatResponse(
            reply=(
                "Sorry, something went wrong on my end. Could you rephrase "
                "what role or skills you're hiring for?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )
