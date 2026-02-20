import pytest 

from time import time 

from tools import create_documentation_tools_cached
from doc_agent import (
    create_agent,
    DocumentationAgentConfig,
    DEFAULT_INSTRUCTIONS,
    AgentStreamRunner
)

from jaxn import JSONParserHandler

from tests.utils import collect_tools, ToolCall


@pytest.fixture(scope="module")
def agent():
    t0 = time()

    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig(
        instructions=DEFAULT_INSTRUCTIONS
    )

    agent = create_agent(agent_config, tools)

    t1 = time()
    print(f'loading agent took {t1 - t0}')

    return agent


async def run_agent_test(agent, user_prompt, message_history=None):
    runner = AgentStreamRunner(agent, JSONParserHandler())
    return await runner.run(user_prompt, message_history)


@pytest.mark.asyncio
async def test_agent_runs(agent):
    user_prompt = 'llm as a judge'
    result = await run_agent_test(agent, user_prompt)

    search_result = result.output
    assert search_result.answer is not None
    assert search_result.confidence >= 0.0
    assert search_result.found_answer is True
    assert len(search_result.followup_questions) > 0


@pytest.mark.asyncio
async def test_agent_uses_tools(agent):
    user_prompt = 'llm as a judge'
    result = await run_agent_test(agent, user_prompt)

    messages = result.new_messages()

    tool_calls = collect_tools(messages)
    assert len(tool_calls) >= 2 

    search_call = tool_calls[0]
    assert search_call.name == 'search'

    get_file_call = tool_calls[1]
    assert get_file_call.name == 'get_file'