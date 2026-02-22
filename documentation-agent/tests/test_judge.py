import pytest 

from tools import create_documentation_tools_cached
from doc_agent import create_agent, DocumentationAgentConfig

from tests.utils import run_agent_test
from tests.judge import assert_criteria


@pytest.fixture(scope="module")
def agent():
    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()
    return create_agent(agent_config, tools)


@pytest.mark.asyncio
async def test_agent_uses_tools(agent):
    user_prompt = 'llm as a judge'
    result = await run_agent_test(agent, user_prompt)

    await assert_criteria(result, [
        "makes at least 2 tool calls",
        "performs search using the 'search' tool",
        "checks the content of the 'examples/LLM_judge.mdx' using 'get_file' tool",
    ])
