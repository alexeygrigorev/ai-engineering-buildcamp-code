import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from logs.loaders import parse_period
from logs.sql import SQLiteStorage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# ── Cost constants ────────────────────────────────────────────────────────────

MODEL_PRICES = {
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "openai:gpt-5.2": {"input": 1.75, "output": 14.00},
    "anthropic:claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "anthropic:claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


def calculate_cost(model_name, input_tokens, output_tokens):
    prices = MODEL_PRICES.get(model_name.lower(), {})
    if not prices:
        return 0.0
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


def count_tool_calls(messages):
    return sum(
        1
        for msg in messages
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    )


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")
st.title("🚀 AI Engineering Monitoring Dashboard")

# ── Session state init ────────────────────────────────────────────────────────

if "selected_session_id" not in st.session_state:
    st.session_state["selected_session_id"] = None
if "events_page" not in st.session_state:
    st.session_state["events_page"] = 0

# ── Sidebar filters ───────────────────────────────────────────────────────────

st.sidebar.header("Time Period")
period_option = st.sidebar.selectbox(
    "Select Period",
    options=["last_30_minutes", "last_hour", "today", "custom"],
    index=2,
)

if period_option == "custom":
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=7))
    start_time_input = col1.time_input("Start Time", datetime.min.time())
    end_date = col2.date_input("End Date", datetime.now())
    end_time_input = col2.time_input("End Time", datetime.now().time())
    start_dt = datetime.combine(start_date, start_time_input)
    end_dt = datetime.combine(end_date, end_time_input)
    period = (start_dt.timestamp(), end_dt.timestamp())
else:
    period = period_option

if st.sidebar.button("🔄 Refresh"):
    st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────

storage = SQLiteStorage()
logs = storage.load_logs(period=period)
events = storage.load_events(period=period)

# Index events by session for fast lookup
events_by_session: dict[str, list] = {}
for e in events:
    events_by_session.setdefault(e.session_id, []).append(e)

# Index logs by session for fast lookup
logs_by_session: dict[str, list] = {}
for l in logs:
    logs_by_session.setdefault(l.session_id, []).append(l)

# Set of sessions that have feedback events
sessions_with_feedback: dict[str, list[int]] = {}
# Dictionary to store evaluation data by session
evaluations_by_session: dict[str, dict] = {}

for e in events:
    if e.event_type == "user_feedback":
        val = e.event_data.get("feedback", e.event_data.get("value", 0))
        sessions_with_feedback.setdefault(e.session_id, []).append(
            1 if val > 0 else -1
        )
    elif e.event_type == "evaluation":
        # We store the latest evaluation for each session
        evaluations_by_session[e.session_id] = e.event_data

# Build full logs DataFrame
if logs:
    df_logs = pd.DataFrame(
        [
            {
                "_idx": i,
                "timestamp": datetime.fromtimestamp(l.timestamp),
                "session_id": l.session_id,
                "model": l.agent_info.model,
                "input_tokens": l.usage.input_tokens,
                "output_tokens": l.usage.output_tokens,
                "total_tokens": l.usage.total_tokens,
                "cost": calculate_cost(
                    l.agent_info.model, l.usage.input_tokens, l.usage.output_tokens
                ),
                "execution_time": l.execution_time,
                "ttft": l.time_to_first_token,
                "tool_calls": count_tool_calls(l.messages),
                "session_convs": len(logs_by_session.get(l.session_id, [])),
                "has_feedback": l.session_id in sessions_with_feedback,
                "feedback_positive": any(
                    v > 0 for v in sessions_with_feedback.get(l.session_id, [])
                ),
                "feedback_negative": any(
                    v < 0 for v in sessions_with_feedback.get(l.session_id, [])
                ),
                "eval_score": evaluations_by_session.get(l.session_id, {}).get("overall_score"),
                "eval_cost": evaluations_by_session.get(l.session_id, {}).get("cost", 0.0),
            }
            for i, l in enumerate(logs)
        ]
    )
    df_logs = df_logs.sort_values("timestamp", ascending=False).reset_index(drop=True)
else:
    df_logs = pd.DataFrame()

if events:
    df_events = pd.DataFrame(
        [
            {
                "timestamp": datetime.fromtimestamp(e.timestamp),
                "session_id": e.session_id,
                "event_type": e.event_type,
                "event_data": json.dumps(e.event_data),
            }
            for e in events
        ]
    ).sort_values("timestamp", ascending=False).reset_index(drop=True)
    
    # Pre-process evaluations for charting
    eval_rows = []
    for e in events:
        if e.event_type == "evaluation":
            data = e.event_data
            row = {
                "timestamp": datetime.fromtimestamp(e.timestamp),
                "session_id": e.session_id,
                "overall_score": data.get("overall_score", 0),
                "cost": data.get("cost", 0),
            }
            # Flatten criteria
            for crit in data.get("criteria", []):
                row[f"crit_{crit['name']}"] = crit.get("score", 0)
            eval_rows.append(row)
    df_evals = pd.DataFrame(eval_rows) if eval_rows else pd.DataFrame()
else:
    df_events = pd.DataFrame()
    df_evals = pd.DataFrame()


# ── Conversation renderer ─────────────────────────────────────────────────────


def render_conversation(messages, final_output=None):
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    with st.expander("🔧 System Prompt", expanded=False):
                        st.code(part.content, language=None, wrap_lines=True)
                elif isinstance(part, UserPromptPart):
                    with st.chat_message("user"):
                        st.markdown(
                            part.content if isinstance(part.content, str) else str(part.content)
                        )
                elif isinstance(part, ToolReturnPart):
                    with st.expander(f"📤 Tool return: **{part.tool_name}**", expanded=False):
                        try:
                            st.json(json.loads(part.content))
                        except Exception:
                            st.code(part.content, language=None, wrap_lines=True)
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    with st.chat_message("assistant"):
                        st.markdown(part.content)
                elif isinstance(part, ToolCallPart):
                    # Hide the raw final_result tool call since we render it beautifully 
                    # at the very end of the conversation via `final_output`.
                    if part.tool_name == "final_result":
                        continue

                    with st.expander(
                        f"🔨 Tool call: **{part.tool_name}**",
                        expanded=False,
                    ):
                        raw = part.args
                        try:
                            st.json(json.loads(raw) if isinstance(raw, str) else raw)
                        except Exception:
                            st.code(str(raw), language=None, wrap_lines=True)
                        
    # The final structured response isn't in the message list natively for Pydantic AI, 
    # but we pass it directly via `final_output`.
    if final_output:
        with st.chat_message("assistant"):
            try:
                # Try parsing it as JSON (useful for RAGResponse)
                parsed = json.loads(final_output) if isinstance(final_output, str) else final_output
                
                if isinstance(parsed, dict) and "answer" in parsed:
                    st.markdown(parsed["answer"])
                    
                    with st.expander("📝 Details & Metadata", expanded=True):
                        # Top row with key metrics
                        m1, m2, m3 = st.columns(3)
                        conf = parsed.get("confidence", 0)
                        conf_color = "green" if conf > 0.8 else "orange" if conf > 0.5 else "red"
                        m1.markdown(f"**Confidence:** :{conf_color}[{conf*100:.0f}%]")
                        m2.markdown(f"**Type:** `{parsed.get('answer_type', 'N/A')}`")
                        m3.markdown(f"**Found Info:** {'✅' if parsed.get('found_answer') else '❌'}")
                        
                        if parsed.get("confidence_explanation"):
                            st.caption(f"*Why:* {parsed['confidence_explanation']}")

                        # References section
                        refs = parsed.get("references", [])
                        if refs:
                            st.markdown("---")
                            st.markdown("**📚 References**")
                            for ref in refs:
                                # Handle both 'file_path' and 'file_name' keys as they appear in different versions
                                fname = ref.get("file_path", ref.get("file_name", "Unknown source"))
                                expl = ref.get("explanation", "")
                                st.markdown(f"📄 `{fname}` — {expl}")

                        # Follow-up questions
                        followups = parsed.get("followup_questions", [])
                        if followups:
                            st.markdown("---")
                            st.markdown("**💡 Suggested Follow-ups**")
                            for q in followups:
                                st.markdown(f"- {q}")
                                
                        # Self-checks
                        checks = parsed.get("checks", [])
                        if checks:
                            with st.expander("✅ Self-Verification Checks", expanded=False):
                                for c in checks:
                                    status = "✅" if c.get("passed") else "❌"
                                    st.markdown(f"{status} **{c.get('rule')}**")
                                    if c.get("explanation"):
                                        st.caption(c["explanation"])
                else:
                    st.json(parsed)
            except Exception as e:
                # Fallback to plain text
                st.markdown(str(final_output))
                st.error(f"Error rendering metadata: {e}")


def render_session(session_id: str):
    """Render the full session explorer for a given session_id."""
    session_logs = sorted(
        logs_by_session.get(session_id, []), key=lambda l: l.timestamp
    )
    session_events = sorted(
        events_by_session.get(session_id, []), key=lambda e: e.timestamp
    )

    if not session_logs:
        st.warning(f"No logs found for session `{session_id}`.")
        return

    total_cost = sum(
        calculate_cost(l.agent_info.model, l.usage.input_tokens, l.usage.output_tokens)
        for l in session_logs
    )

    st.subheader(f"📦 Session `{session_id}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversations", len(session_logs))
    c2.metric("Events", len(session_events))
    c3.metric("Total Cost", f"${total_cost:.4f}")
    c4.metric("Total Tokens", f"{sum(l.usage.total_tokens for l in session_logs):,}")

    # Events timeline
    if session_events:
        st.markdown("**Events timeline:**")
        ev_rows = [
            {
                "time": datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S"),
                "type": e.event_type,
                "data": json.dumps(e.event_data),
            }
            for e in session_events
        ]
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Each conversation
    for i, log_record in enumerate(session_logs):
        ai = log_record.agent_info
        conv_cost = calculate_cost(ai.model, log_record.usage.input_tokens, log_record.usage.output_tokens)
        label = (
            f"💬 Conversation {i + 1} — "
            f"{ai.model.split(':')[-1]} · "
            f"{log_record.usage.total_tokens:,} tokens · "
            f"${conv_cost:.4f} · "
            f"{log_record.execution_time:.1f}s"
        )
        with st.expander(label, expanded=(i == 0)):
            m1, m2, m3 = st.columns(3)
            m1.caption(f"**TTFT:** {log_record.time_to_first_token:.2f}s")
            m2.caption(f"**In:** {log_record.usage.input_tokens:,} / **Out:** {log_record.usage.output_tokens:,} tokens")
            m3.caption(f"**Tool calls:** {count_tool_calls(log_record.messages)}")

            if ai.tools:
                st.markdown("**Tools:** " + " · ".join(f"`{t}`" for t in ai.tools))
            st.markdown("---")
            
            # Show evaluation if available
            eval_data = evaluations_by_session.get(session_id)
            if eval_data and i == len(session_logs) - 1:
                with st.expander("⭐ **Judge Evaluation**", expanded=True):
                    sc1, sc2, sc3 = st.columns([1, 1, 1])
                    score = eval_data.get('overall_score', 0)
                    color = "green" if score > 0.8 else "orange" if score > 0.5 else "red"
                    sc1.metric("Overall Score", f"{score*100:.0f}%", delta=None)
                    sc2.metric("Eval Cost", f"${eval_data.get('cost', 0):.4f}")
                    
                    st.markdown(f"**Summary:** {eval_data.get('summary', 'No summary')}")
                    
                    # Criteria table
                    criteria = eval_data.get('criteria', [])
                    if criteria:
                        crit_df = pd.DataFrame(criteria)
                        # Format criteria table
                        crit_df['status'] = crit_df['passed'].apply(lambda x: "✅" if x else "❌")
                        crit_df = crit_df[['status', 'name', 'score', 'reasoning']]
                        st.table(crit_df)
                    
                    if eval_data.get('improvement_suggestions'):
                        st.info(f"💡 **Suggestions:** {eval_data['improvement_suggestions']}")
                st.markdown("---")
            
            render_conversation(log_record.messages, final_output=log_record.output)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_logs, tab_events, tab_evals, tab_session = st.tabs(
    ["📊 Overview", "📋 Logs", "🔔 Events", "📉 Evaluations", "🔬 Session Explorer"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    if df_logs.empty:
        st.info("No logs found for the selected period.")
    else:
        total_cost = df_logs["cost"].sum()
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("💰 Total Cost", f"${total_cost:.4f}")
        c2.metric("🔢 Total Tokens", f"{int(df_logs['total_tokens'].sum()):,}")
        c3.metric("📨 Requests", len(df_logs))
        c4.metric("🛠 Tool Calls", int(df_logs["tool_calls"].sum()))
        c5.metric("⏱ Avg Exec Time", f"{df_logs['execution_time'].mean():.2f}s")
        c6.metric("⚡ Avg TTFT", f"{df_logs['ttft'].mean():.2f}s")

        # Evaluation metrics row
        st.divider()
        st.subheader("🤖 Automated Evaluations")
        e1, e2, e3, e4, _ = st.columns([1, 1, 1, 1, 2])
        eval_logs = df_logs[df_logs["eval_score"].notnull()]
        if not eval_logs.empty:
            avg_score = eval_logs["eval_score"].mean()
            total_eval_cost = eval_logs["eval_cost"].sum()
            e1.metric("⭐ Avg Eval Score", f"{avg_score:.2f}")
            e2.metric("📊 Evaluated", f"{len(eval_logs)} sessions")
            e3.metric("📈 Coverage", f"{len(eval_logs)/len(df_logs)*100:.0f}%")
            e4.metric("💸 Eval Cost", f"${total_eval_cost:.4f}")
        else:
            st.info("No evaluations found for this period.")

        st.divider()
        st.subheader("By model")
        model_stats = (
            df_logs.groupby("model")
            .agg(requests=("model", "count"), total_tokens=("total_tokens", "sum"), total_cost=("cost", "sum"), avg_exec_time=("execution_time", "mean"))
            .reset_index()
        )
        model_stats["total_cost"] = model_stats["total_cost"].map("${:.4f}".format)
        model_stats["avg_exec_time"] = model_stats["avg_exec_time"].map("{:.2f}s".format)
        model_stats["total_tokens"] = model_stats["total_tokens"].map("{:,}".format)
        st.dataframe(model_stats, use_container_width=True, hide_index=True)

    if not df_events.empty:
        feedback = df_events[df_events["event_type"] == "user_feedback"]
        if not feedback.empty:
            st.divider()
            st.subheader("User Feedback")
            parsed_vals = feedback["event_data"].apply(
                lambda d: json.loads(d) if isinstance(d, str) else d
            ).apply(lambda d: d.get("feedback", d.get("value", 0)))
            fa, fb, _ = st.columns([1, 1, 4])
            fa.metric("👍 Positive", int((parsed_vals > 0).sum()))
            fb.metric("👎 Negative", int((parsed_vals < 0).sum()))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Logs + filtering
# ─────────────────────────────────────────────────────────────────────────────
with tab_logs:
    if df_logs.empty:
        st.info("No logs found for the selected period.")
    else:
        # ── Filters ───────────────────────────────────────────────────────
        with st.expander("🔎 Filter logs", expanded=False):
            fc1, fc2, fc3 = st.columns(3)

            all_models = sorted(df_logs["model"].unique().tolist())
            sel_models = fc1.multiselect("Model", all_models, default=all_models)

            min_convs = fc2.slider(
                "Min conversations in session", 1, max(int(df_logs["session_convs"].max()), 2), 1
            )
            min_tools = fc2.slider(
                "Min tool calls", 0, max(int(df_logs["tool_calls"].max()), 1), 0
            )

            feedback_filter = fc3.radio(
                "Feedback",
                ["All", "👍 Thumbs up", "👎 Thumbs down", "Any feedback"],
                index=0,
            )

        # Apply filters
        mask = (
            df_logs["model"].isin(sel_models)
            & (df_logs["session_convs"] >= min_convs)
            & (df_logs["tool_calls"] >= min_tools)
        )
        if feedback_filter == "👍 Thumbs up":
            mask &= df_logs["feedback_positive"]
        elif feedback_filter == "👎 Thumbs down":
            mask &= df_logs["feedback_negative"]
        elif feedback_filter == "Any feedback":
            mask &= df_logs["has_feedback"]

        df_filtered = df_logs[mask]
        st.caption(f"Showing {len(df_filtered)} of {len(df_logs)} log records")

        # ── Table with View buttons ────────────────────────────────────────
        header = st.columns([2, 1.5, 1.5, 1, 1, 1, 1, 1, 1])
        for h, label in zip(header, ["Timestamp", "Session", "Model", "Tokens", "Cost", "Exec", "Tools", "Eval", ""]):
            h.markdown(f"**{label}**")
        st.divider()

        for _, row in df_filtered.iterrows():
            cols = st.columns([2, 1.5, 1.5, 1, 1, 1, 1, 1, 1])
            cols[0].caption(row["timestamp"].strftime("%m-%d %H:%M:%S"))
            cols[1].caption(row["session_id"][:12] + "…")
            cols[2].caption(row["model"].split(":")[-1])
            cols[3].caption(f"{int(row['total_tokens']):,}")
            cols[4].caption(f"${row['cost']:.4f}")
            cols[5].caption(f"{row['execution_time']:.2f}s")
            cols[6].caption(str(int(row["tool_calls"])))
            
            # Eval Score column
            score = row["eval_score"]
            if score is not None:
                color = "green" if score > 0.8 else "orange" if score > 0.5 else "red"
                cols[7].markdown(f":{color}[**{score:.2f}**]")
            else:
                cols[7].caption("—")

            if cols[8].button("🔬", key=f"view_{row['_idx']}_{row['session_id'][:8]}"):
                st.session_state["selected_session_id"] = row["session_id"]
                st.toast("✅ Session loaded — open the **Session Explorer** tab", icon="🔬")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Events with search + pagination
# ─────────────────────────────────────────────────────────────────────────────
with tab_events:
    if df_events.empty:
        st.info("No events found for the selected period.")
    else:
        # Summary counts
        evt_counts = df_events.groupby("event_type").size().reset_index(name="count").sort_values("count", ascending=False)
        cols = st.columns(min(len(evt_counts), 6))
        for col, (_, row) in zip(cols, evt_counts.iterrows()):
            col.metric(row["event_type"], int(row["count"]))

        st.divider()

        # Search + filter
        sc1, sc2 = st.columns([2, 2])
        search_text = sc1.text_input("🔍 Search (type or data)", placeholder="e.g. followup, feedback…")
        all_event_types = sorted(df_events["event_type"].unique().tolist())
        sel_event_types = sc2.multiselect("Event types", all_event_types, default=all_event_types)

        # Apply search filter
        ef = df_events[df_events["event_type"].isin(sel_event_types)]
        if search_text:
            q = search_text.lower()
            ef = ef[
                ef["event_type"].str.lower().str.contains(q)
                | ef["event_data"].str.lower().str.contains(q)
                | ef["session_id"].str.lower().str.contains(q)
            ]

        # Pagination
        PAGE_SIZE = 50
        total_pages = max(1, (len(ef) + PAGE_SIZE - 1) // PAGE_SIZE)
        page = st.session_state.get("events_page", 0)
        page = max(0, min(page, total_pages - 1))

        p1, p2, p3 = st.columns([1, 3, 1])
        if p1.button("◀ Prev", disabled=(page == 0)):
            st.session_state["events_page"] = page - 1
            st.rerun()
        p2.markdown(
            f"<div style='text-align:center;padding-top:6px'>Page {page + 1} of {total_pages} &nbsp;·&nbsp; {len(ef)} events</div>",
            unsafe_allow_html=True,
        )
        if p3.button("Next ▶", disabled=(page >= total_pages - 1)):
            st.session_state["events_page"] = page + 1
            st.rerun()

        page_slice = ef.iloc[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
        st.dataframe(
            page_slice[["timestamp", "session_id", "event_type", "event_data"]],
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Evaluations
# ─────────────────────────────────────────────────────────────────────────────
with tab_evals:
    if df_evals.empty:
        st.info("No evaluations found for the selected period.")
    else:
        st.subheader("🤖 Automated Evaluation Quality Trends")
        
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Quality Score", f"{df_evals['overall_score'].mean()*100:.1f}%")
        m2.metric("Evaluations Count", len(df_evals))
        m3.metric("Total Eval Cost", f"${df_evals['cost'].sum():.4f}")
        m4.metric("Success Rate (Score > 0.8)", f"{(df_evals['overall_score'] > 0.8).mean()*100:.0f}%")
        
        st.divider()
        
        # Charts row
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Overall Score Distribution**")
            # Create a histogram using pandas/streamlit
            hist_df = df_evals['overall_score'].round(1).value_counts().sort_index().reset_index()
            hist_df.columns = ['score', 'count']
            st.bar_chart(hist_df.set_index('score'))
            
        with c2:
            st.markdown("**Criteria Performance**")
            # Average score per criteria
            crit_cols = [c for c in df_evals.columns if c.startswith("crit_")]
            if crit_cols:
                avg_crit = df_evals[crit_cols].mean().reset_index()
                avg_crit.columns = ['Criterion', 'Average Score']
                avg_crit['Criterion'] = avg_crit['Criterion'].str.replace("crit_", "")
                st.bar_chart(avg_crit.set_index('Criterion'))
            else:
                st.caption("No individual criteria data available.")
        
        st.divider()
        st.subheader("📋 Recent Evaluations")
        
        # Table of evaluations with view buttons
        eval_display = df_evals.copy()
        eval_display = eval_display.sort_values("timestamp", ascending=False)
        
        header = st.columns([2, 3, 1, 1, 1])
        header[0].markdown("**Timestamp**")
        header[1].markdown("**Session ID**")
        header[2].markdown("**Score**")
        header[3].markdown("**Cost**")
        header[4].markdown("")
        
        for idx, row in eval_display.head(20).iterrows():
            cols = st.columns([2, 3, 1, 1, 1])
            cols[0].caption(row["timestamp"].strftime("%m-%d %H:%M:%S"))
            cols[1].caption(row["session_id"])
            
            score = row["overall_score"]
            color = "green" if score > 0.8 else "orange" if score > 0.5 else "red"
            cols[2].markdown(f":{color}[**{score:.2f}**]")
            cols[3].caption(f"${row['cost']:.4f}")
            
            if cols[4].button("🔬 View", key=f"eval_view_{idx}_{row['session_id'][:8]}"):
                st.session_state["selected_session_id"] = row["session_id"]
                st.toast("✅ Session loaded — switching to Explorer", icon="🔬")
                # We can't easily switch tabs from code in Streamlit without query params or a complex session state, 
                # but setting the ID is enough for the user to click over.


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Session Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_session:
    sel_sid = st.session_state.get("selected_session_id")

    # Also allow manual session selection
    all_session_ids = sorted(set(l.session_id for l in logs), reverse=True)
    if all_session_ids:
        default_idx = all_session_ids.index(sel_sid) if sel_sid in all_session_ids else 0
        chosen = st.selectbox(
            "Session ID",
            options=all_session_ids,
            index=default_idx,
            format_func=lambda s: s[:20] + "…" + f"  ({len(logs_by_session.get(s, []))} conv, {len(events_by_session.get(s, []))} events)",
        )
        if chosen != sel_sid:
            st.session_state["selected_session_id"] = chosen
            sel_sid = chosen

    if sel_sid:
        st.divider()
        render_session(sel_sid)
    else:
        st.info("Click 🔬 on a log row in the **Logs** tab, or pick a session ID above.")
