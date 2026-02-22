
from typing import Dict, Any
from dataclasses import dataclass

from tools import create_documentation_tools_cached
from doc_agent import (
    create_agent,
    AgentStreamRunner
)

from jaxn import JSONParserHandler

from tests.cost_tracker import capture_usage


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


def collect_tools(messages):
    tool_calls = []

    for m in messages:
        for p in m.parts:
            part_kind = p.part_kind

            if part_kind != 'tool-call':
                continue

            if p.tool_name == 'final_result':
                continue

            tool_calls.append(ToolCall(p.tool_name, p.args))

    return tool_calls


def get_model_name(agent):
    provider = agent.model.system
    model_name = agent.model.model_name
    return f'{provider}:{model_name}'


async def run_agent_test(agent, user_prompt, message_history=None):
    runner = AgentStreamRunner(agent, JSONParserHandler())
    result = await runner.run(user_prompt, message_history)

    model = get_model_name(agent)
    capture_usage(model, result)

    return result