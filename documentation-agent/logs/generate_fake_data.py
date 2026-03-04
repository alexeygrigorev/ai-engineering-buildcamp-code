"""
Fake data generator for the monitoring dashboard.

Generates realistic user interactions with the documentation agent at ~2 events/second.
Each "session" simulates a user asking questions, clicking follow-ups, and leaving feedback.

Usage:
    uv run python generate_fake_data.py
    uv run python generate_fake_data.py --rate 2.0 --sessions 5
"""

import argparse
import json
import random
import time
from uuid import uuid4

from logs.models import AgentInfo, LogEvent, LogRecord
from logs.sql import SQLiteStorage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai import RunUsage

# ── Fake content pools ────────────────────────────────────────────────────────

QUESTIONS = [
    "How do I create a custom metric in Evidently?",
    "What is the difference between a DataDrift and TargetDrift test?",
    "How do I run a test suite on a pandas DataFrame?",
    "Can I use Evidently with Spark?",
    "How do I add Evidently to a CI/CD pipeline?",
    "What reports are available out of the box?",
    "How do I track model performance over time?",
    "What is a Reference Dataset and why do I need one?",
    "How do I log predictions in production?",
    "Can Evidently detect concept drift?",
    "How do I set custom thresholds for tests?",
    "What database backends does Evidently support?",
    "How do I compare two models side by side?",
    "What is the TextDescriptors preset?",
    "How do I schedule monitoring jobs?",
]

FOLLOW_UPS = [
    "Can you give me a code example?",
    "How does this work with streaming data?",
    "What are the performance implications?",
    "Is there a simpler alternative?",
    "How do I configure this in YAML?",
    "Does this work with scikit-learn models?",
    "What version was this introduced in?",
    "Are there any known limitations?",
]

ANSWER_TYPES = ["how-to", "explanation", "troubleshooting", "comparison", "reference"]

DOC_FILES = [
    "docs/monitoring/custom-metrics.md",
    "docs/tests/test-suites.md",
    "docs/integrations/spark.md",
    "docs/reports/data-drift.md",
    "docs/reports/target-drift.md",
    "docs/production/logging.md",
    "docs/tests/thresholds.md",
    "docs/reports/text.md",
    "docs/monitoring/scheduling.md",
]

MODELS = [
    "openai:gpt-4o-mini",
    "openai:gpt-4o",
    "anthropic:claude-haiku-4-5",
]

AGENT_INSTRUCTIONS = (
    "You are a helpful assistant for the Evidently AI documentation. "
    "Answer questions about data quality, model monitoring, and test suites."
)

TOOLS = ["search", "get_file", "final_result"]


# ── Message builder helpers ───────────────────────────────────────────────────


def make_fake_messages(question: str, answer: str) -> list:
    """Build a realistic pydantic-ai message history for a single agent turn."""
    search_query = question[:50]
    doc_file = random.choice(DOC_FILES)

    return [
        ModelRequest(
            parts=[
                SystemPromptPart(content=AGENT_INSTRUCTIONS),
                UserPromptPart(content=question),
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="search",
                    args=json.dumps({"query": search_query}),
                    tool_call_id="tc1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="search",
                    content=f"Found relevant docs: {doc_file}",
                    tool_call_id="tc1",
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="get_file",
                    args=json.dumps({"filename": doc_file}),
                    tool_call_id="tc2",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="get_file",
                    content="# Documentation content\n" + answer[:200],
                    tool_call_id="tc2",
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args=make_fake_output(question),
                    tool_call_id="tc3",
                )
            ]
        ),
    ]


def make_fake_output(question: str) -> str:
    """Build a JSON string that resembles a RAGResponse."""
    answer = f"To address your question about '{question[:40]}...', here is the relevant information from the Evidently documentation."
    doc_file = random.choice(DOC_FILES)
    follow_ups = random.sample(FOLLOW_UPS, k=3)
    answer_type = random.choice(ANSWER_TYPES)
    confidence = round(random.uniform(0.55, 0.98), 2)

    resp = {
        "answer": answer,
        "found_answer": random.random() > 0.15,
        "references": [
            {
                "file_path": doc_file,
                "explanation": "Primary source for this topic.",
            }
        ],
        "confidence": confidence,
        "confidence_explanation": f"The documentation directly covers this topic with {'high' if confidence > 0.75 else 'moderate'} specificity.",
        "answer_type": answer_type,
        "followup_questions": follow_ups,
        "checks": [{"rule": "Factual accuracy", "passed": True, "explanation": "Verified against docs."}],
    }
    return json.dumps(resp)


# ── Core generators ───────────────────────────────────────────────────────────


def generate_log_record(session_id: str, model: str, question: str) -> LogRecord:
    answer = f"Here is the information about: {question}"
    messages = make_fake_messages(question, answer)
    output = messages[-1].parts[0].args  # The RAGResponse JSON generated above

    input_tokens = random.randint(300, 2000)
    output_tokens = random.randint(150, 800)

    return LogRecord(
        session_id=session_id,
        timestamp=time.time(),
        agent_info=AgentInfo(
            name="documentation-agent",
            model=model,
            instructions=AGENT_INSTRUCTIONS,
            tools=TOOLS,
        ),
        messages=messages,
        usage=RunUsage(
            requests=random.randint(1, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        time_to_first_token=round(random.uniform(0.3, 2.5), 3),
        execution_time=round(random.uniform(1.0, 8.0), 3),
        output=output,
    )


def generate_event(session_id: str, event_type: str, event_data: dict) -> LogEvent:
    return LogEvent(
        session_id=session_id,
        timestamp=time.time(),
        event_type=event_type,
        event_data=event_data,
    )


# ── Session simulation ────────────────────────────────────────────────────────


def simulate_session(storage: SQLiteStorage, rate: float, verbose: bool = True):
    """
    Simulate a single user session:
    1. User asks a question  → LogRecord
    2. User clicks follow-up → LogEvent (followup_clicked)
    3. User asks follow-up   → LogRecord
    4. User leaves feedback  → LogEvent (user_feedback)
    """
    session_id = str(uuid4())
    model = random.choice(MODELS)
    delay = 1.0 / rate  # seconds between events

    def emit_log(question: str):
        record = generate_log_record(session_id, model, question)
        storage.save_log(record)
        if verbose:
            print(f"  [LOG]   session={session_id[:8]}  model={model}  q='{question[:45]}...'")
        time.sleep(delay)

    def emit_event(event_type: str, data: dict):
        event = generate_event(session_id, event_type, data)
        storage.save_event(event)
        if verbose:
            print(f"  [EVENT] session={session_id[:8]}  type={event_type}  data={data}")
        time.sleep(delay)

    question = random.choice(QUESTIONS)
    emit_log(question)

    # Sometimes the user clicks a follow-up
    if random.random() > 0.3:
        followup_q = random.choice(FOLLOW_UPS)
        emit_event("followup_clicked", {"question": followup_q})

        # And asks it
        if random.random() > 0.4:
            emit_log(followup_q)

    # Sometimes leaves feedback
    if random.random() > 0.4:
        feedback_value = random.choices([1, -1], weights=[0.75, 0.25])[0]
        emit_event("user_feedback", {"feedback": feedback_value})


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate fake monitoring data for the dashboard."
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="Events per second (default: 2.0)",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=0,
        help="Number of sessions to generate then stop (0 = run forever, default: 0)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="db/logs.db",
        help="Path to the SQLite database (default: db/logs.db)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-event output",
    )
    args = parser.parse_args()

    storage = SQLiteStorage(db_path=args.db)
    verbose = not args.quiet
    session_count = 0

    print(f"🚀 Fake data generator started")
    print(f"   rate    : {args.rate} events/sec")
    print(f"   sessions: {'∞ (Ctrl-C to stop)' if args.sessions == 0 else args.sessions}")
    print(f"   db      : {args.db}")
    print()

    try:
        while True:
            session_count += 1
            if verbose:
                print(f"── Session #{session_count} ──")
            simulate_session(storage, rate=args.rate, verbose=verbose)

            if args.sessions > 0 and session_count >= args.sessions:
                break

            # Short pause between sessions
            time.sleep(1.0 / args.rate)

    except KeyboardInterrupt:
        print(f"\n✅ Done. Generated {session_count} sessions.")


if __name__ == "__main__":
    main()
