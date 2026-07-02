"""
Replays the real user turns from the 10 provided traces through the
TestClient-mounted pipeline (FakeEmbedder + MockLLMClient).

As explained in tests/conftest.py, FakeEmbedder isn't semantically
meaningful, so these tests check PLUMBING correctness across every real
conversation shape we were given (multi-turn refine sequences, compare
questions, confirmation phrases, long JD-style messages, etc.) -- every
turn must produce a schema-valid response and the pipeline must never
crash. Actual Recall@10 against these same traces, using the real embedding
model, is measured by scripts/run_eval.py against a running deployment.
"""
import re
from pathlib import Path

import pytest

TRACES_DIR = Path(__file__).parent.parent / "traces"
TURN_USER_RE = re.compile(r"\*\*User\*\*\s*\n\n>\s*(.+?)\s*\n\n\*\*Agent\*\*", re.DOTALL)


def _load_all_traces():
    traces = []
    for path in sorted(TRACES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        user_messages = [m.strip() for m in TURN_USER_RE.findall(text)]
        if user_messages:
            traces.append((path.stem, user_messages))
    return traces


TRACES = _load_all_traces()


@pytest.mark.parametrize("name,user_messages", TRACES, ids=[t[0] for t in TRACES])
def test_trace_replay_never_crashes_and_stays_schema_valid(client, name, user_messages):
    history = []
    for turn_num, user_msg in enumerate(user_messages, start=1):
        history.append({"role": "user", "content": user_msg})
        resp = client.post("/chat", json={"messages": history})

        assert resp.status_code == 200, f"{name} turn {turn_num} returned {resp.status_code}"
        data = resp.json()

        assert isinstance(data["reply"], str) and data["reply"], (
            f"{name} turn {turn_num}: empty reply"
        )
        assert 0 <= len(data["recommendations"]) <= 10, (
            f"{name} turn {turn_num}: recommendations out of bounds"
        )
        assert isinstance(data["end_of_conversation"], bool)

        history.append({"role": "assistant", "content": data["reply"]})


def test_provided_traces_are_consistent_with_the_8_exchange_interpretation():
    """The assignment says the evaluator caps conversations at '8 turns
    including user & assistant'. Taken as 8 total messages, this would be
    violated by the provided traces themselves: C3 has 10 total messages
    and C9 has 14. Since these are SHL's own ground-truth examples, we
    trust that evidence over the ambiguous wording and interpret the cap as
    8 user-assistant EXCHANGES (up to 16 total messages) -- see
    app/config.py MAX_TOTAL_MESSAGES for the full reasoning. This test just
    documents that every provided trace fits comfortably under that
    corrected interpretation, with room to spare."""
    for name, user_messages in TRACES:
        total_messages = len(user_messages) * 2
        assert total_messages <= 16, (
            f"{name} has {total_messages} messages, exceeds even the corrected "
            f"16-message (8-exchange) interpretation -- re-check the cap reading"
        )
