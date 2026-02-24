"""Roundtrip tests: PydanticAI messages -> OTel format -> back to PydanticAI messages.

Uses PydanticAI's built-in `messages_to_otel_messages()` as the forward converter,
then our `otel_to_model_messages()` as the reverse, and checks content is preserved.
"""

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

from trace_replay.converter import _extract_output, otel_to_model_messages


def to_otel(messages: list[ModelMessage]) -> list[dict]:
    """Convert PydanticAI messages to OTel format using the official converter."""
    settings = InstrumentationSettings()
    return settings.messages_to_otel_messages(messages)


def roundtrip(messages: list[ModelMessage]) -> list[ModelMessage]:
    """PydanticAI -> OTel -> PydanticAI roundtrip."""
    otel = to_otel(messages)
    return otel_to_model_messages(otel)


# --- Basic message types ---


def test_simple_text_roundtrip():
    """User text in, assistant text out."""
    original = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi there")]),
    ]
    result = roundtrip(original)

    assert len(result) == 2
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[1], ModelResponse)

    assert isinstance(result[0].parts[0], UserPromptPart)
    assert result[0].parts[0].content == "hello"

    assert isinstance(result[1].parts[0], TextPart)
    assert result[1].parts[0].content == "hi there"


def test_system_prompt_roundtrip():
    """System prompt + user prompt should roundtrip.

    OTel splits system and user into separate OTel messages, but our converter
    merges consecutive non-assistant messages back into a single ModelRequest.
    """
    original = [
        ModelRequest(parts=[
            SystemPromptPart(content="You are helpful."),
            UserPromptPart(content="hello"),
        ]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]
    result = roundtrip(original)

    assert len(result) == 2
    assert isinstance(result[0], ModelRequest)
    assert len(result[0].parts) == 2
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == "You are helpful."
    assert isinstance(result[0].parts[1], UserPromptPart)
    assert result[0].parts[1].content == "hello"

    assert isinstance(result[1], ModelResponse)
    assert result[1].parts[0].content == "hi"


# --- Tool calls ---


def test_tool_call_roundtrip():
    """Tool call + tool return should roundtrip."""
    original = [
        ModelRequest(parts=[UserPromptPart(content="search for cats")]),
        ModelResponse(parts=[
            ToolCallPart(
                tool_name="search",
                args={"query": "cats"},
                tool_call_id="call_123",
            ),
        ], finish_reason="tool_call"),
        ModelRequest(parts=[
            ToolReturnPart(
                tool_name="search",
                content={"results": ["cat1", "cat2"]},
                tool_call_id="call_123",
            ),
        ]),
        ModelResponse(parts=[TextPart(content="I found 2 cats.")]),
    ]
    result = roundtrip(original)

    assert len(result) == 4

    # Check tool call
    resp1 = result[1]
    assert isinstance(resp1, ModelResponse)
    assert isinstance(resp1.parts[0], ToolCallPart)
    assert resp1.parts[0].tool_name == "search"
    assert resp1.parts[0].args == {"query": "cats"}
    assert resp1.parts[0].tool_call_id == "call_123"

    # Check tool return
    req2 = result[2]
    assert isinstance(req2, ModelRequest)
    assert isinstance(req2.parts[0], ToolReturnPart)
    assert req2.parts[0].tool_name == "search"
    assert req2.parts[0].content == {"results": ["cat1", "cat2"]}
    assert req2.parts[0].tool_call_id == "call_123"

    # Check final text
    assert result[3].parts[0].content == "I found 2 cats."


def test_multiple_tool_calls_in_one_response():
    """Multiple tool calls in a single assistant response."""
    original = [
        ModelRequest(parts=[UserPromptPart(content="compare weather")]),
        ModelResponse(parts=[
            ToolCallPart(tool_name="weather", args={"city": "NYC"}, tool_call_id="c1"),
            ToolCallPart(tool_name="weather", args={"city": "LA"}, tool_call_id="c2"),
        ], finish_reason="tool_call"),
        ModelRequest(parts=[
            ToolReturnPart(tool_name="weather", content="sunny", tool_call_id="c1"),
            ToolReturnPart(tool_name="weather", content="rainy", tool_call_id="c2"),
        ]),
        ModelResponse(parts=[TextPart(content="NYC is sunny, LA is rainy.")]),
    ]
    result = roundtrip(original)

    assert len(result) == 4
    assert len(result[1].parts) == 2
    assert result[1].parts[0].tool_name == "weather"
    assert result[1].parts[1].tool_name == "weather"
    assert len(result[2].parts) == 2


def test_thinking_roundtrip():
    """ThinkingPart should survive the roundtrip."""
    original = [
        ModelRequest(parts=[UserPromptPart(content="think hard")]),
        ModelResponse(parts=[
            ThinkingPart(content="Let me think about this..."),
            TextPart(content="Here's my answer."),
        ]),
    ]
    result = roundtrip(original)

    assert len(result) == 2
    resp = result[1]
    assert isinstance(resp, ModelResponse)
    assert len(resp.parts) == 2
    assert isinstance(resp.parts[0], ThinkingPart)
    assert resp.parts[0].content == "Let me think about this..."
    assert isinstance(resp.parts[1], TextPart)
    assert resp.parts[1].content == "Here's my answer."


# --- Agent integration tests ---


@pytest.mark.anyio
async def test_agent_roundtrip():
    """Run an actual agent with TestModel, roundtrip its messages."""
    agent = Agent(model=TestModel(custom_output_text="test response"), system_prompt="Be helpful.")
    result = await agent.run("hello")
    original_messages = result.all_messages()

    otel = to_otel(original_messages)
    recovered = otel_to_model_messages(otel)

    # Verify response text matches
    original_responses = [m for m in original_messages if isinstance(m, ModelResponse)]
    recovered_responses = [m for m in recovered if isinstance(m, ModelResponse)]
    assert len(original_responses) == len(recovered_responses)

    for orig, rec in zip(original_responses, recovered_responses):
        orig_texts = [p.content for p in orig.parts if isinstance(p, TextPart)]
        rec_texts = [p.content for p in rec.parts if isinstance(p, TextPart)]
        assert orig_texts == rec_texts

    # Verify user prompts match
    original_user = [
        p.content for m in original_messages if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, UserPromptPart)
    ]
    recovered_user = [
        p.content for m in recovered if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, UserPromptPart)
    ]
    assert original_user == recovered_user

    # Verify system prompts match
    original_system = [
        p.content for m in original_messages if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, SystemPromptPart)
    ]
    recovered_system = [
        p.content for m in recovered if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, SystemPromptPart)
    ]
    assert original_system == recovered_system


@pytest.mark.anyio
async def test_agent_with_tools_roundtrip():
    """Run an agent with tools, roundtrip its messages."""
    agent = Agent(model=TestModel(), system_prompt="Use tools.")

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    result = await agent.run("greet Alice")
    original_messages = result.all_messages()

    otel = to_otel(original_messages)
    recovered = otel_to_model_messages(otel)

    # Verify tool calls survived
    original_tool_calls = [
        (p.tool_name, p.tool_call_id)
        for m in original_messages if isinstance(m, ModelResponse)
        for p in m.parts if isinstance(p, ToolCallPart)
    ]
    recovered_tool_calls = [
        (p.tool_name, p.tool_call_id)
        for m in recovered if isinstance(m, ModelResponse)
        for p in m.parts if isinstance(p, ToolCallPart)
    ]
    assert original_tool_calls == recovered_tool_calls

    # Verify tool returns survived
    original_returns = [
        (p.tool_name, p.tool_call_id, p.content)
        for m in original_messages if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, ToolReturnPart)
    ]
    recovered_returns = [
        (p.tool_name, p.tool_call_id, p.content)
        for m in recovered if isinstance(m, ModelRequest)
        for p in m.parts if isinstance(p, ToolReturnPart)
    ]
    assert original_returns == recovered_returns


# --- Edge cases ---


def test_finish_reason_preserved():
    """finish_reason on ModelResponse should survive roundtrip."""
    original = [
        ModelRequest(parts=[UserPromptPart(content="hi")]),
        ModelResponse(parts=[TextPart(content="bye")], finish_reason="stop"),
    ]
    result = roundtrip(original)
    assert result[1].finish_reason == "stop"


def test_tool_call_with_string_args():
    """Tool call args as JSON string (not dict) should roundtrip.

    String args get parsed to dict during OTel conversion, so we get dict back.
    """
    original = [
        ModelRequest(parts=[UserPromptPart(content="go")]),
        ModelResponse(parts=[
            ToolCallPart(
                tool_name="search",
                args='{"query": "test"}',
                tool_call_id="c1",
            ),
        ]),
    ]
    result = roundtrip(original)

    tool_call = result[1].parts[0]
    assert isinstance(tool_call, ToolCallPart)
    assert tool_call.args == {"query": "test"}


def test_empty_conversation():
    """Empty message list should roundtrip to empty."""
    assert roundtrip([]) == []


def test_multi_turn_conversation():
    """Multi-turn conversation with interleaved tool calls."""
    original = [
        ModelRequest(parts=[
            SystemPromptPart(content="You are a search assistant."),
            UserPromptPart(content="find info about Python"),
        ]),
        ModelResponse(parts=[
            ToolCallPart(tool_name="search", args={"q": "Python"}, tool_call_id="c1"),
        ], finish_reason="tool_call"),
        ModelRequest(parts=[
            ToolReturnPart(tool_name="search", content="Python is a language", tool_call_id="c1"),
        ]),
        ModelResponse(parts=[
            TextPart(content="Based on my search, Python is a programming language."),
        ], finish_reason="stop"),
        ModelRequest(parts=[UserPromptPart(content="tell me more")]),
        ModelResponse(parts=[
            ToolCallPart(tool_name="search", args={"q": "Python details"}, tool_call_id="c2"),
        ], finish_reason="tool_call"),
        ModelRequest(parts=[
            ToolReturnPart(tool_name="search", content="Python was created by Guido", tool_call_id="c2"),
        ]),
        ModelResponse(parts=[
            TextPart(content="Python was created by Guido van Rossum."),
        ], finish_reason="stop"),
    ]
    result = roundtrip(original)

    all_text = [
        p.content for m in result if isinstance(m, ModelResponse)
        for p in m.parts if isinstance(p, TextPart)
    ]
    assert all_text == [
        "Based on my search, Python is a programming language.",
        "Python was created by Guido van Rossum.",
    ]

    all_tool_names = [
        p.tool_name for m in result if isinstance(m, ModelResponse)
        for p in m.parts if isinstance(p, ToolCallPart)
    ]
    assert all_tool_names == ["search", "search"]


# --- Output extraction ---


def test_extract_output_text():
    """Text output: last ModelResponse with TextPart."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="hi")]),
        ModelResponse(parts=[TextPart(content="hello there")]),
    ]
    assert _extract_output(messages) == "hello there"


def test_extract_output_structured():
    """Structured output: last ModelResponse is a tool call (e.g. final_result)."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="search")]),
        ModelResponse(parts=[
            ToolCallPart(
                tool_name="final_result",
                args={"answer": "42", "confidence": 0.9},
                tool_call_id="c1",
            ),
        ]),
        ModelRequest(parts=[
            ToolReturnPart(tool_name="final_result", content="OK", tool_call_id="c1"),
        ]),
    ]
    output = _extract_output(messages)
    assert isinstance(output, dict)
    assert output == {"answer": "42", "confidence": 0.9}


def test_extract_output_structured_with_tools_before():
    """Structured output after intermediate tool calls."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="find cats")]),
        ModelResponse(parts=[
            ToolCallPart(tool_name="search", args={"q": "cats"}, tool_call_id="c1"),
        ]),
        ModelRequest(parts=[
            ToolReturnPart(tool_name="search", content="found cats", tool_call_id="c1"),
        ]),
        ModelResponse(parts=[
            ToolCallPart(
                tool_name="final_result",
                args={"answer": "cats found", "refs": ["a.md"]},
                tool_call_id="c2",
            ),
        ]),
        ModelRequest(parts=[
            ToolReturnPart(tool_name="final_result", content="done", tool_call_id="c2"),
        ]),
    ]
    output = _extract_output(messages)
    assert isinstance(output, dict)
    assert output["answer"] == "cats found"


def test_extract_output_empty():
    """Empty messages returns empty string."""
    assert _extract_output([]) == ""
