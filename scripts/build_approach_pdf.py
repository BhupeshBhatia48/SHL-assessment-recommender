from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

GREEN = colors.HexColor("#2E7D32")
DARK = colors.HexColor("#1a1a1a")
GRAY = colors.HexColor("#555555")
BORDER = colors.HexColor("#dddddd")

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleCustom", parent=styles["Title"], fontSize=16, textColor=DARK,
    spaceAfter=4, leading=19,
)
subtitle_style = ParagraphStyle(
    "SubtitleCustom", parent=styles["Normal"], fontSize=9, textColor=GRAY,
    spaceAfter=10, leading=12,
)
h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=11, textColor=GREEN,
    spaceBefore=10, spaceAfter=4, leading=13,
)
body = ParagraphStyle(
    "BodyCustom", parent=styles["Normal"], fontSize=8.7, textColor=DARK,
    leading=12.2, spaceAfter=6, alignment=4,  # justify
)

def hr():
    return HRFlowable(width="100%", thickness=0.6, color=BORDER, spaceBefore=2, spaceAfter=8)

doc = SimpleDocTemplate(
    "/home/claude/shl-assessment-recommender/APPROACH.pdf",
    pagesize=letter,
    topMargin=0.6 * inch, bottomMargin=0.6 * inch,
    leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    title="Approach Document - SHL Conversational Assessment Recommender",
)

story = []
story.append(Paragraph("Approach Document", title_style))
story.append(Paragraph("SHL Conversational Assessment Recommender &mdash; AI Intern Take-Home Assignment", subtitle_style))
story.append(hr())

story.append(Paragraph("Design choices", h2))
story.append(Paragraph(
    "The system is a retrieval-grounded pipeline, not a multi-agent framework: guardrail pre-check &rarr; "
    "analyzer (LLM call) &rarr; router &rarr; retrieval/compare &rarr; responder (LLM call) &rarr; validator. "
    "FastAPI was chosen for native Pydantic validation matching the required schema exactly; no database or "
    "session store is used since the API is contractually stateless &mdash; every <font face='Courier'>/chat</font> "
    "call re-derives everything it needs from the full message history it's given.",
    body,
))
story.append(Paragraph(
    "The single most important decision: the LLM never free-types a name or URL. The responder only selects "
    "assessment <b>ids</b> from a closed list of retrieved candidates; the actual <font face='Courier'>recommendations</font> "
    "JSON is assembled afterward directly from our own validated catalog records, then cross-checked against the "
    "live catalog again in a final validator. This makes hallucinated names/URLs structurally impossible rather "
    "than merely discouraged by prompting, and is the main lever against the hard-eval scoring component.",
    body,
))

story.append(Paragraph("Retrieval setup", h2))
story.append(Paragraph(
    "The provided catalog (377 Individual Test Solutions) was normalized into a clean schema: name, url, "
    "test_type (single-letter code derived from category tags), description, job_levels, languages, duration. "
    "Two real data-quality bugs were caught and fixed by cross-referencing the 10 provided traces against the raw "
    "scrape: (1) one product name was corrupted by a stray control character mid-scrape &mdash; repaired narrowly "
    "for that one record, after a broader &ldquo;rebuild from URL slug&rdquo; heuristic was tested against the full "
    "catalog and found to corrupt 22 otherwise-correct names (e.g. &ldquo;.NET Framework 4.5&rdquo; &rarr; &ldquo;Net "
    "Framework 4 5&rdquo;) and was scrapped; (2) multi-category &ldquo;report&rdquo; products collapse to Test Type "
    "&ldquo;D&rdquo; rather than joining every tag, matching the one confirmed example in the traces.",
    body,
))
story.append(Paragraph(
    "Retrieval uses Google's hosted text-embedding-004 model via the Gemini API over name + categories + job "
    "levels + description, indexed with FAISS IndexFlatIP (exact cosine search &mdash; at 377 items, approximate "
    "indexing adds complexity with no benefit). The query is built each turn from the LLM-extracted constraint "
    "set, not the raw user message, so retrieval benefits from accumulated context rather than just the latest "
    "sentence. A local sentence-transformers backend was tried first and remains available in the codebase for "
    "self-hosting with more memory &mdash; see Evaluation approach below for why the default moved to the hosted API.",
    body,
))

story.append(Paragraph("Prompt design", h2))
story.append(Paragraph(
    "Two LLM calls per turn, deliberately narrow in responsibility. The <b>analyzer</b> re-derives intent and the "
    "full constraint set from the entire conversation history on every call (not a diff of the newest message) "
    "&mdash; this is what makes &ldquo;refine&rdquo; work with zero special-case code: a later message that adds or "
    "changes a constraint just merges into the same extraction step every time. The <b>responder</b> only selects "
    "from a candidate list handed to it and writes the reply text; it cannot introduce new names. A separate prompt "
    "handles <b>compare</b>, grounded strictly in the specific catalog records resolved for the named assessments "
    "(exact match, falling back to fuzzy matching for close variants like &ldquo;OPQ&rdquo; vs &ldquo;OPQ32r&rdquo;).",
    body,
))
story.append(Paragraph(
    "A code-level guardrail pre-check (regex patterns for injection, legal advice, general hiring advice, and "
    "clearly off-topic content) runs before any LLM call, so the clearest refusal cases are instant and don't "
    "depend on the model choosing to refuse gracefully.",
    body,
))

story.append(Paragraph("Evaluation approach", h2))
story.append(Paragraph(
    "<font face='Courier'>scripts/run_eval.py</font> parses the 10 provided trace files (extracting each trace's "
    "user turns and its final markdown table as ground truth), replays the user turns against a live "
    "<font face='Courier'>/chat</font> endpoint, and computes Recall@10 by comparing predicted vs. expected URLs. "
    "All ground-truth URLs referenced across the 10 traces were confirmed to exist in the normalized catalog "
    "before any pipeline code was written.",
    body,
))
story.append(Paragraph(
    "<b>What didn't work / limitation to flag honestly:</b> the assignment states the evaluator caps conversations "
    "at &ldquo;8 turns including user &amp; assistant.&rdquo; Taken literally as 8 total messages, this is "
    "contradicted by the traces themselves &mdash; trace C9 has 7 exchanges (14 messages) and C3 has 10. Since "
    "these are SHL's own ground-truth examples, that evidence was trusted over the ambiguous wording, and the cap "
    "was implemented as 8 exchanges (up to 16 messages), with a hard override: on the last available exchange, the "
    "router forces a best-effort shortlist rather than risk ending on an unanswered clarifying question with zero "
    "recall.",
    body,
))
story.append(Paragraph(
    "A more consequential lesson: the original retrieval design used a local sentence-transformers model. That "
    "worked locally, but the first real deployment to Render's free tier failed outright with &ldquo;Ran out of "
    "memory (used over 512MB)&rdquo; &mdash; PyTorch's footprint alone is close to that ceiling. Rather than paying "
    "for a larger instance, the embedding backend was swapped to a hosted Gemini text-embedding-004 API call, "
    "reusing the same key already required for chat completions. This required zero changes to the retrieval logic "
    "itself, because it was built from the start against a pluggable Embedder interface rather than a specific "
    "backend &mdash; the swap was a one-line change at the call site. This is exactly the kind of real deployment "
    "constraint that's easy to miss when developing locally and worth designing around up front.",
    body,
))
story.append(Paragraph(
    "A third limitation: the development sandbox used to build this had no internet access to HuggingFace, so "
    "retrieval quality was validated locally with a TF-IDF stand-in (keyword-only) rather than a real embedding "
    "model, and the pytest suite uses a deterministic fake embedder for the same reason &mdash; both clearly scoped "
    "as dev/test-only in code comments. The actual deployment runs the real Gemini embedding API; real Recall@10 "
    "numbers should be captured with <font face='Courier'>run_eval.py</font> against the live deployment before "
    "final submission.",
    body,
))

story.append(Paragraph("AI tool disclosure", h2))
story.append(Paragraph(
    "This solution was built in collaboration with Claude (Anthropic), used for architecture discussion, code "
    "generation across all pipeline modules, catalog data-quality debugging (including catching and reverting a "
    "name-corruption regression before it shipped), and drafting this document. All design decisions and trade-offs "
    "above were reviewed and are defensible in the technical interview.",
    body,
))

doc.build(story)
print("Approach PDF built")
