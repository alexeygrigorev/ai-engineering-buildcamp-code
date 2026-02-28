import asyncio
import json

import streamlit as st
from jaxn import JSONParserHandler, StreamingJSONParser
from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent

from doc_agent import DocumentationAgentConfig, create_agent
from models import RAGResponse
from tools import create_documentation_tools_cached

import dotenv

dotenv.load_dotenv()


GITHUB_BASE = "https://github.com/evidentlyai/docs/blob/main/"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evidently Docs Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
<style>
/* Dark, clean base */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 0.5rem;
}

/* Activity expander */
.activity-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0 0.8rem 0;
    font-size: 0.82rem;
    color: #8b949e;
}
.activity-item {
    padding: 2px 0;
}
.activity-item a {
    color: #58a6ff;
    text-decoration: none;
}
.activity-item a:hover {
    text-decoration: underline;
}

/* Metadata badges */
.meta-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
}
.badge-type   { border-color: #388bfd; color: #79c0ff; }
.badge-conf   { border-color: #3fb950; color: #56d364; }
.badge-found  { border-color: #f78166; color: #ffa198; }
.badge-found-yes { border-color: #3fb950; color: #56d364; }

/* References section */
.references-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #c9d1d9;
    margin: 1rem 0 0.5rem 0;
}
.reference-item {
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 0.4rem;
    line-height: 1.4;
}
.reference-item a {
    color: #58a6ff;
    text-decoration: none;
    font-weight: 500;
}
.reference-item a:hover {
    text-decoration: underline;
}

/* Follow-up section */
.followup-label {
    font-size: 0.78rem;
    color: #8b949e;
    margin-bottom: 0.3rem;
}

/* Feedback buttons styling - more specific selector to avoid collisions */
div[data-testid="column"] .stButton button:has(div:contains("👍")),
div[data-testid="column"] .stButton button:has(div:contains("👎")) {
    border-radius: 20px !important;
    padding: 0px 8px !important;
    min-height: 24px !important;
    height: 24px !important;
    line-height: 24px !important;
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-size: 0.8rem !important;
    margin-top: 5px !important;
}
div[data-testid="column"] .stButton button:has(div:contains("👍")):hover,
div[data-testid="column"] .stButton button:has(div:contains("👎")):hover {
    border-color: #58a6ff !important;
    color: #58a6ff !important;
    background: #30363d !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session-state init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, meta, activities
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []  # pydantic-ai message history
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None


# ── Load agent (cached per session) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    tools = create_documentation_tools_cached()
    config = DocumentationAgentConfig(
        # model="openai:gpt-5.2"
        # model="anthropic:claude-sonnet-4-6"
        model="openai:gpt-4o-mini"
    )
    agent = create_agent(config, tools)
    return agent


with st.spinner("Loading index…"):
    agent: Agent = load_agent()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Docs Assistant")
    st.caption("Powered by Evidently AI documentation")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_messages = []
        st.session_state.pending_followup = None
        st.rerun()


# ── Helpers ──────────────────────────────────────────────────────────────────
def render_activities(activities: list[str], status: str):
    """Render the agent activity panel."""
    label = "⚙️ Agent activity — " + status
    lines = "\n".join(f'<div class="activity-item">{a}</div>' for a in activities)
    html = f'<div class="activity-box"><strong>{label}</strong><br>{lines}</div>'
    return html


def render_metadata(meta: dict) -> str:
    answer_type = meta.get("answer_type", "—")
    confidence = meta.get("confidence")
    found = meta.get("found_answer")

    conf_str = f"{confidence * 100:.0f}%" if confidence is not None else "—"
    found_cls = "badge-found-yes" if found else "badge-found"
    found_str = "Found: Yes" if found else "Found: No"

    return (
        f'<div class="meta-row" style="display: flex; align-items: center; gap: 0.5rem;">'
        f'<span class="badge badge-type">🏷 {answer_type}</span>'
        f'<span class="badge badge-conf">🎯 {conf_str}</span>'
        f'<span class="badge {found_cls}">{found_str}</span>'
        f"</div>"
    )


def render_references(references: list[dict]) -> str:
    if not references:
        return ""
    html = '<div class="references-label">📚 References</div>'
    for ref in references:
        file_name = ref.get("file_name", ref.get("file_path", "unknown"))
        explanation = ref.get("explanation", "")
        url = GITHUB_BASE + file_name
        html += f'<div class="reference-item">📄 <a href="{url}" target="_blank">{file_name}</a> — {explanation}</div>'
    return html


def activity_html_for_tool(tool_name: str, args_str: str) -> str:
    """Format a single tool call as an HTML activity line."""
    try:
        args = json.loads(args_str) if args_str else {}
    except Exception:
        args = {}

    if tool_name == "search":
        query = args.get("query", args_str)
        return f'🔍 <em>Search:</em> "{query}"'
    elif tool_name == "get_file":
        filename = args.get("filename", args_str)
        url = GITHUB_BASE + filename
        return f'📄 <em>File:</em> <a href="{url}" target="_blank">{filename}</a>'
    elif tool_name == "final_result":
        return "✅ <em>Generating answer…</em>"
    else:
        return f"⚙️ {tool_name}({args_str[:60]})"


# ── Streaming runner ─────────────────────────────────────────────────────────
class UIStreamHandler(JSONParserHandler):
    """Captures streaming JSON chunks for the answer field."""

    def __init__(self, text_placeholder, ref_placeholder=None):
        self._placeholder = text_placeholder
        self._ref_placeholder = ref_placeholder
        self._answer_so_far = ""
        self.metadata = {}
        self.followup_questions = []
        self.references = []

    def on_value_chunk(self, path: str, field_name: str, chunk: str) -> None:
        if path == "" and field_name == "answer":
            self._answer_so_far += chunk
            self._placeholder.markdown(self._answer_so_far + " ▌")

    def on_field_end(
        self, path: str, field_name: str, value: str, parsed_value=None
    ) -> None:
        if path == "":
            if field_name == "answer":
                self._placeholder.markdown(self._answer_so_far)
            elif field_name in (
                "answer_type",
                "found_answer",
                "confidence",
                "confidence_explanation",
            ):
                try:
                    self.metadata[field_name] = json.loads(value)
                except Exception:
                    self.metadata[field_name] = value

    def on_array_item_end(self, path: str, field_name: str, item=None) -> None:
        if field_name == "followup_questions" and item is not None:
            self.followup_questions.append(item)
        elif field_name == "references" and item is not None:
            self.references.append(item)
            if self._ref_placeholder:
                self._ref_placeholder.markdown(render_references(self.references), unsafe_allow_html=True)

    @property
    def answer(self) -> str:
        return self._answer_so_far


async def run_streaming(user_prompt: str, message_history: list, activities_ref: list):
    """
    Run the agent with streaming.
    Returns (answer, metadata_dict, followup_questions, references, new_messages, act_list).
    """
    answer = ""
    metadata = {}
    followup_questions = []
    references = []
    new_messages = []

    # We need a placeholder for streaming text — we create it in the caller
    # and pass it via a mutable container.
    # Instead, we yield events through a queue consumed by the Streamlit layer.
    # For simplicity we run the coroutine directly (Streamlit ≥ 1.30 allows asyncio.run).

    # Placeholders injected from outside via closure
    text_ph = activities_ref[0]  # text placeholder for the answer
    act_ph = activities_ref[1]  # placeholder for activity panel
    ref_ph = activities_ref[2] if len(activities_ref) > 2 else None
    act_list = []  # accumulates activity items

    def _update_activities(status: str):
        act_ph.markdown(render_activities(act_list, status), unsafe_allow_html=True)

    _update_activities("Thinking…")

    parser = StreamingJSONParser(UIStreamHandler(text_ph, ref_ph))
    handler = parser.handler  # type: UIStreamHandler

    args_so_far = ""

    async with agent.iter(
        user_prompt,
        message_history=message_history,
        output_type=RAGResponse,
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node):
                args_so_far = ""
                async with node.stream(agent_run.ctx) as stream:
                    async for response in stream.stream_responses():
                        for part in response.parts:
                            if part.part_kind != "tool-call":
                                continue
                            if part.tool_name == "final_result":
                                new_chunk = part.args[len(args_so_far) :]
                                args_so_far = part.args
                                parser.parse_incremental(new_chunk)
                            # Skip other tool calls here — args are incomplete
                            # during streaming; they are handled in CallToolsNode.

            elif Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as events:
                    async for event in events:
                        if isinstance(event, FunctionToolCallEvent):
                            tool_name = event.part.tool_name
                            raw_args = event.part.args
                            args_str = (
                                raw_args
                                if isinstance(raw_args, str)
                                else json.dumps(raw_args)
                            )
                            act_item = activity_html_for_tool(tool_name, args_str)
                            if act_item not in act_list and tool_name != "final_result":
                                act_list.append(act_item)
                                _update_activities("Thinking…")

        new_messages = agent_run.result.new_messages() if agent_run.result else []

    _update_activities("Done ✓")

    answer = handler.answer
    metadata = handler.metadata
    followup_questions = handler.followup_questions
    references = handler.references

    return answer, metadata, followup_questions, references, new_messages, act_list


# ── Render existing chat history ─────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Activity panel above the answer (mirrors live streaming layout)
            if msg.get("activities"):
                st.markdown(
                    render_activities(msg["activities"], "Done ✓"),
                    unsafe_allow_html=True,
                )
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # References below the answer
            if msg.get("references"):
                st.markdown(render_references(msg["references"]), unsafe_allow_html=True)
            # Metadata and Feedback below the answer
            meta_col, feedback_col = st.columns([0.8, 0.2])
            with meta_col:
                if msg.get("meta"):
                    st.markdown(render_metadata(msg["meta"]), unsafe_allow_html=True)

            with feedback_col:
                f1, f2 = st.columns(2)
                if f1.button("👍", key=f"upvote_{idx}"):
                    st.toast("Thanks for the feedback!", icon="👍")
                if f2.button("👎", key=f"downvote_{idx}"):
                    st.toast("Thanks for the feedback!", icon="👎")


# ── Follow-up buttons (only for last assistant message) ─────────────────────
last_followups = []
for msg in reversed(st.session_state.messages):
    if msg["role"] == "assistant" and msg.get("followup_questions"):
        last_followups = msg["followup_questions"]
        break

if last_followups and st.session_state.pending_followup is None:
    st.markdown(
        '<div class="followup-label">💡 Suggested follow-ups</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(last_followups))
    for col, q in zip(cols, last_followups):
        if col.button(q, key=f"followup_{q[:40]}"):
            with logfire.attach_context(st.session_state.logfire_context):
                logfire.info("followup_question_clicked", question=q)
            st.session_state.pending_followup = q
            st.rerun()


# ── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about Evidently documentation…")

# Resolve prompt: typed input OR pending follow-up
prompt = None
if user_input:
    prompt = user_input
    st.session_state.pending_followup = None
elif st.session_state.pending_followup:
    prompt = st.session_state.pending_followup
    st.session_state.pending_followup = None


# ── Handle new prompt ────────────────────────────────────────────────────────
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show assistant response shell
    with st.chat_message("assistant"):
        act_placeholder = st.empty()  # activity panel
        answer_placeholder = st.empty()  # streaming answer
        ref_placeholder = st.empty()  # references panel

        # Run the agent
        answer, metadata, followup_questions, references, new_messages, act_list = asyncio.run(
            run_streaming(
                prompt,
                st.session_state.agent_messages,
                [answer_placeholder, act_placeholder, ref_placeholder],
            )
        )

        # Render final metadata and feedback
        if metadata:
            meta_col, feedback_col = st.columns([0.8, 0.2])
            with meta_col:
                st.markdown(render_metadata(metadata), unsafe_allow_html=True)
            with feedback_col:
                f1, f2 = st.columns(2)
                cur_idx = len(st.session_state.messages)
                if f1.button("👍", key=f"upvote_live"):
                    with logfire.attach_context(st.session_state.logfire_context):
                        logfire.info("user_feedback", feedback=1)
                    st.toast("Thanks for the feedback!", icon="👍")
                if f2.button("👎", key=f"downvote_live"):
                    with logfire.attach_context(st.session_state.logfire_context):
                        logfire.info("user_feedback", feedback=-1)
                    st.toast("Thanks for the feedback!", icon="👎")

    # Persist to session
    st.session_state.agent_messages.extend(new_messages)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "activities": act_list,
            "references": references,
            "meta": metadata,
            "followup_questions": followup_questions,
        }
    )

    st.rerun()
