"""
Local evaluation harness against the 10 provided conversation traces.

WHAT THIS DOES
Each trace file (traces/C1.md ... C10.md) is a markdown transcript with a
"### Turn N" per exchange, a quoted **User** message, an **Agent** reply, and
(when the agent has committed to a shortlist) a markdown table of
recommended assessments with a URL column. We treat the LAST such table in
the file as the trace's ground-truth final shortlist.

For each trace, we replay ONLY the recorded user messages, in order, against
our own live POST /chat endpoint -- our agent's own replies (not the
trace's scripted agent replies) become the conversation history for
subsequent turns, exactly like the real evaluator would do. After the final
user turn, we compare our agent's final `recommendations` (by URL) against
the trace's ground-truth URL set and compute Recall@10.

IMPORTANT LIMITATION (documented honestly): the real SHL evaluator uses a
*simulated user LLM* that can adapt its answers to whatever our agent asks,
whereas this script replays static, pre-written user messages. If our agent
asks a different clarifying question than the original trace's agent did,
the next scripted user message might not perfectly answer it. This is a
reasonable and useful local proxy -- not a perfect reproduction of the real
harness -- and is exactly the kind of gap worth calling out explicitly in
the approach document's "what didn't work / how we measured it" section.

USAGE
    python scripts/run_eval.py                  # uses http://localhost:8000
    python scripts/run_eval.py --url https://your-deployed-url.onrender.com
"""
import argparse
import re
import sys
from pathlib import Path

import requests

TRACES_DIR = Path(__file__).parent.parent / "traces"

TURN_USER_RE = re.compile(r"\*\*User\*\*\s*\n\n>\s*(.+?)\s*\n\n\*\*Agent\*\*", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|.*?\|\s*<?(https?://\S+?)>?\s*\|\s*$", re.MULTILINE)


def parse_trace(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    user_messages = [m.strip() for m in TURN_USER_RE.findall(text)]

    # Ground truth = the LAST markdown table's URL column in the file.
    all_rows = TABLE_ROW_RE.findall(text)
    # TABLE_ROW_RE matches across the whole file; we only want rows from the
    # last table block. Split the file on blank-line-separated blocks that
    # contain a header row, and take the rows following the last header.
    header_positions = [m.start() for m in re.finditer(r"^\|\s*#\s*\|", text, re.MULTILINE)]
    if header_positions:
        last_header_pos = header_positions[-1]
        tail = text[last_header_pos:]
        expected_urls = [url.strip() for _, url in TABLE_ROW_RE.findall(tail)]
    else:
        expected_urls = []

    return {"name": path.stem, "user_messages": user_messages, "expected_urls": expected_urls}


def replay_trace(trace: dict, base_url: str) -> dict:
    history = []
    last_response = None
    for user_msg in trace["user_messages"]:
        history.append({"role": "user", "content": user_msg})
        resp = requests.post(f"{base_url}/chat", json={"messages": history}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        last_response = data
        history.append({"role": "assistant", "content": data.get("reply", "")})

    predicted_urls = [r["url"] for r in (last_response or {}).get("recommendations", [])]
    expected = set(trace["expected_urls"])
    predicted = set(predicted_urls)

    if not expected:
        recall = None
    else:
        recall = len(expected & predicted) / len(expected)

    return {
        "name": trace["name"],
        "expected": expected,
        "predicted": predicted,
        "recall_at_10": recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    trace_files = sorted(TRACES_DIR.glob("*.md"))
    if not trace_files:
        print(f"No trace files found in {TRACES_DIR}")
        sys.exit(1)

    results = []
    for path in trace_files:
        trace = parse_trace(path)
        if not trace["user_messages"]:
            print(f"WARNING: no user turns parsed from {path.name}, skipping")
            continue
        try:
            result = replay_trace(trace, args.url)
        except requests.RequestException as e:
            print(f"ERROR replaying {path.name}: {e}")
            continue
        results.append(result)

        recall_str = f"{result['recall_at_10']:.2f}" if result["recall_at_10"] is not None else "N/A"
        print(f"\n{result['name']}: Recall@10 = {recall_str}")
        print(f"  expected : {sorted(result['expected'])}")
        print(f"  predicted: {sorted(result['predicted'])}")
        missed = result["expected"] - result["predicted"]
        if missed:
            print(f"  MISSED: {sorted(missed)}")

    scored = [r["recall_at_10"] for r in results if r["recall_at_10"] is not None]
    if scored:
        mean_recall = sum(scored) / len(scored)
        print(f"\n{'=' * 50}")
        print(f"Mean Recall@10 across {len(scored)} traces: {mean_recall:.3f}")
    else:
        print("\nNo scoreable traces (no ground-truth URLs parsed).")


if __name__ == "__main__":
    main()
