import pytest 

from tools import create_documentation_tools_cached
from doc_agent import (
    create_agent,
    run_agent_stream,
    DocumentationAgentConfig,
    DEFAULT_INSTRUCTIONS
)

def create_test_agent():
    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig(
        instructions=DEFAULT_INSTRUCTIONS
    )

    agent = create_agent(agent_config, tools)
    return agent


@pytest.mark.asyncio
async def test_agent_runs():
    agent = create_test_agent()

    user_prompt = 'llm as a judge'    
    result = await run_agent_stream(agent, user_prompt)

    search_result = result.output
    assert search_result.answer is not None
    assert search_result.confidence >= 0.0
    assert search_result.found_answer is True
    assert len(search_result.followup_questions) > 0
