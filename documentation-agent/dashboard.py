import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from logs.sql import SQLiteStorage
from logs.loaders import parse_period
from pydantic_ai.messages import ModelResponse, ToolCallPart

# Cost constants from tests/cost_tracker.py
MODEL_PRICES = {
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "openai:gpt-5.2": {"input": 1.75, "output": 14.00},
    "anthropic:claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "anthropic:claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

def calculate_cost(model_name, input_tokens, output_tokens):
    model_name = model_name.lower()
    if model_name not in MODEL_PRICES:
        return 0.0
    prices = MODEL_PRICES[model_name]
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost

def count_tool_calls(messages):
    count = 0
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    count += 1
    return count

st.set_page_config(page_title="Monitoring Dashboard", layout="wide")

st.title("🚀 AI Engineering Monitoring Dashboard")

# Sidebar for period selection
st.sidebar.header("Filter Settings")
period_option = st.sidebar.selectbox(
    "Select Period",
    options=["last_30_minutes", "last_hour", "today", "custom"],
    index=2
)

start_time, end_time = None, None
if period_option == "custom":
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=7))
    start_time_input = col1.time_input("Start Time", datetime.min.time())
    end_date = col2.date_input("End Date", datetime.now())
    end_time_input = col2.time_input("End Time", datetime.now().time())
    
    start_dt = datetime.combine(start_date, start_time_input)
    end_dt = datetime.combine(end_date, end_time_input)
    start_time, end_time = start_dt.timestamp(), end_dt.timestamp()
    period = (start_time, end_time)
else:
    period = period_option

# Load data
storage = SQLiteStorage()
logs = storage.load_logs(period=period)
events = storage.load_events(period=period)

# Data Processing
if logs:
    df_logs = pd.DataFrame([{
        "timestamp": datetime.fromtimestamp(l.timestamp),
        "session_id": l.session_id,
        "model": l.agent_info.model,
        "input_tokens": l.usage.input_tokens,
        "output_tokens": l.usage.output_tokens,
        "total_tokens": l.usage.total_tokens,
        "cost": calculate_cost(l.agent_info.model, l.usage.input_tokens, l.usage.output_tokens),
        "execution_time": l.execution_time,
        "ttft": l.time_to_first_token,
        "tool_calls": count_tool_calls(l.messages)
    } for l in logs])
    df_logs = df_logs.sort_values("timestamp", ascending=False)
else:
    df_logs = pd.DataFrame()

if events:
    df_events = pd.DataFrame([{
        "timestamp": datetime.fromtimestamp(e.timestamp),
        "session_id": e.session_id,
        "event_type": e.event_type,
        "event_data": e.event_data # Keep as dict
    } for e in events])
    df_events = df_events.sort_values("timestamp", ascending=False)
else:
    df_events = pd.DataFrame()

# Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

if not df_logs.empty:
    total_cost = df_logs["cost"].sum()
    total_tokens = df_logs["total_tokens"].sum()
    total_tool_calls = df_logs["tool_calls"].sum()
    avg_exec_time = df_logs["execution_time"].mean()
    avg_ttft = df_logs["ttft"].mean()

    col1.metric("Total Cost", f"${total_cost:.4f}")
    col2.metric("Total Tokens", f"{total_tokens:,}")
    col3.metric("Tool Calls", f"{total_tool_calls}")
    col4.metric("Avg Exec Time", f"{avg_exec_time:.2f}s")
    col5.metric("Avg TTFT", f"{avg_ttft:.2f}s")
else:
    st.info("No logs found for the selected period.")

# Feedback Analysis
if not df_events.empty:
    feedback_events = df_events[df_events["event_type"] == "user_feedback"]
    if not feedback_events.empty:
        st.subheader("User Feedback")
        # Try different possible keys for feedback value
        def get_feedback_value(data):
            # Try 'feedback', then 'value'
            return data.get("feedback", data.get("value", 0))

        feedback_values = feedback_events["event_data"].apply(get_feedback_value)
        pos = (feedback_values > 0).sum()
        neg = (feedback_values < 0).sum()
        
        c1, c2, c3 = st.columns([1, 1, 3])
        c1.metric("👍 Positive", pos)
        c2.metric("👎 Negative", neg)
else:
    st.info("No events found for the selected period.")

# Visualization
if not df_logs.empty:
    st.subheader("Usage Over Time")
    
    # Resample for time-series charts
    # We use a 5-minute window if 'today' is selected, otherwise 1-minute
    freq = "5min" if period_option == "today" else "1min"
    df_ts = df_logs.set_index("timestamp").resample(freq).sum().fillna(0)
    
    if not df_ts.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Token Usage")
            st.line_chart(df_ts[["input_tokens", "output_tokens"]])
        with c2:
            st.write("Estimated Cost")
            st.line_chart(df_ts["cost"])

# Tables
st.subheader("Recent Logs")
if not df_logs.empty:
    # Display a cleaner version of the table
    display_cols = ["timestamp", "model", "total_tokens", "cost", "execution_time", "tool_calls"]
    st.dataframe(df_logs[display_cols], use_container_width=True)
else:
    st.write("No logs available.")

st.subheader("Recent Events")
if not df_events.empty:
    st.dataframe(df_events, use_container_width=True)
else:
    st.write("No events available.")

if st.button("Refresh Data"):
    st.rerun()

