# OTel ↔ PydanticAI Message Converter

## Why this module exists

When PydanticAI agents run with Logfire instrumentation, all conversation messages are stored in OpenTelemetry (OTel) format — a simplified, provider-agnostic representation. This happens automatically via PydanticAI's built-in `InstrumentedModel`.

The problem: this conversion is one-way. PydanticAI has `messages_to_otel_messages()` to convert to OTel format, but no reverse converter to get back to native `ModelMessage` objects. That means data stored in Logfire can't be directly loaded back into PydanticAI for replay, analysis, or continued conversations.

This module provides the missing reverse converter.

## How it works

### The two message formats

PydanticAI native format uses `kind`/`part_kind` discriminators:
```json
[
  {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "hello"}]},
  {"kind": "response", "parts": [{"part_kind": "text", "content": "hi there"}]}
]
```

OTel format (stored in Logfire) uses `role`/`type`:
```json
[
  {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
  {"role": "assistant", "parts": [{"type": "text", "content": "hi there"}]}
]
```

### Forward conversion (PydanticAI → OTel)

Done by PydanticAI internally. The code lives in two places:

1. Orchestrator: `pydantic_ai/models/instrumented.py` — `InstrumentationSettings.messages_to_otel_messages()`
   - Iterates `ModelMessage` objects
   - Splits `ModelRequest` parts into system vs user groups via `itertools.groupby`
   - Wraps `ModelResponse` as `role: "assistant"` with optional `finish_reason`

2. Per-part converters: `pydantic_ai/messages.py` — `.otel_message_parts()` on each part type
   - `SystemPromptPart` → `{"type": "text", "content": "..."}` with `role: "system"`
   - `UserPromptPart` → `{"type": "text", "content": "..."}` with `role: "user"`
   - `ToolReturnPart` → `{"type": "tool_call_response", "id": "...", "name": "...", "result": ...}` with `role: "user"`
   - `TextPart` → `{"type": "text", "content": "..."}` with `role: "assistant"`
   - `ToolCallPart` → `{"type": "tool_call", "id": "...", "name": "...", "arguments": ...}` with `role: "assistant"`
   - `ThinkingPart` → `{"type": "thinking", "content": "..."}` with `role: "assistant"`

### Reverse conversion (OTel → PydanticAI) — this module

`otel_to_model_messages()` reverses the mapping:

| OTel message | PydanticAI type |
|---|---|
| `role: "system"` + `type: "text"` | `SystemPromptPart` in `ModelRequest` |
| `role: "user"` + `type: "text"` | `UserPromptPart` in `ModelRequest` |
| `role: "user"` + `type: "tool_call_response"` | `ToolReturnPart` in `ModelRequest` |
| `role: "assistant"` + `type: "text"` | `TextPart` in `ModelResponse` |
| `role: "assistant"` + `type: "tool_call"` | `ToolCallPart` in `ModelResponse` |
| `role: "assistant"` + `type: "thinking"` | `ThinkingPart` in `ModelResponse` |

Consecutive system/user OTel messages are merged into a single `ModelRequest` (pending parts accumulate until an assistant message flushes them).

### Known limitations

The OTel conversion is lossy — these fields are not preserved:
- `timestamp` on message parts
- `provider_name`, `provider_details`, `provider_response_id` on responses
- `model_name` on responses
- `usage` (token counts) on individual responses — only available as span attributes
- `dynamic_ref` on system prompts
- `run_id`, `metadata` on messages
- System + user parts that were in the same `ModelRequest` come back as separate `ModelRequest` objects

## How Logfire stores the data

Logfire stores PydanticAI data across two span types:

### `chat` spans (per-model-call)
```sql
SELECT
    attributes->'gen_ai.input.messages' as input_messages,
    attributes->'gen_ai.output.messages' as output_messages,
    attributes->>'gen_ai.usage.input_tokens' as input_tokens,
    attributes->>'gen_ai.usage.output_tokens' as output_tokens
FROM records
WHERE trace_id = '...' AND span_name LIKE 'chat%'
ORDER BY start_timestamp ASC
```

### `agent run` spans (per-agent-run)
```sql
SELECT
    attributes->'pydantic_ai.all_messages' as all_messages
FROM records
WHERE trace_id = '...' AND span_name = 'agent run'
```

Both store messages in OTel format (not PydanticAI native format).

Note: despite the name `pydantic_ai.all_messages`, the data is in OTel format, not PydanticAI's native format. This is because PydanticAI converts messages before storing them:

```python
# agent/__init__.py:826
'pydantic_ai.all_messages': json.dumps(settings.messages_to_otel_messages(list(message_history)))
```

The native PydanticAI format (with `kind`/`part_kind` discriminators, produced by `ModelMessagesTypeAdapter.dump_json()`) is never written to any span attribute. This is why a reverse converter is necessary.

## Usage

### Convert OTel messages to PydanticAI format
```python
from trace_replay import otel_to_model_messages

messages = otel_to_model_messages(otel_data)
# Returns list[ModelMessage] — can pass to agent.run(message_history=messages)
```

### Reconstruct AgentRunResult from Logfire chat span rows
```python
from trace_replay import rows_to_run_result

result = rows_to_run_result(rows)
result.output              # final text answer
result.all_messages()      # list[ModelMessage]
result.usage()             # RunUsage with token counts
```

## Tests

```bash
pytest trace_replay/test_converter.py -v
```

Tests use PydanticAI's `InstrumentationSettings.messages_to_otel_messages()` as the forward converter and verify our reverse converter preserves all message content through the roundtrip.
