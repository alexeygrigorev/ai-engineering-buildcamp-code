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


@pytest.mark.asyncio
async def test_ambiguous_term_judge(agent):
    user_prompt = "judge"
    result = await run_agent_test(agent, user_prompt)

    await assert_criteria(result, [
        "the answer is about 'LLM as a judge' (using an LLM to evaluate outputs), not about legal judges or the judicial system",
        "follow-up questions are about LLM evaluation or related ML topics, not about the legal system or courts",
    ])


@pytest.mark.skip(reason="gpt-4o-mini copies code from docs verbatim, ignoring formatting instructions")
@pytest.mark.asyncio
async def test_code_formatting(agent):
    user_prompt = "how to use LLM as a judge"
    result = await run_agent_test(agent, user_prompt)

    await assert_criteria(result, [
        "code examples use 4-space indentation",
        "function/constructor calls with multiple keyword arguments place each argument on its own line",
        "keyword arguments in function calls have no spaces around '=' (e.g. provider=\"openai\", not provider = \"openai\")",
        "normal variable assignments still use spaces around '=' (e.g. llm_eval = LLMEval(...))",
    ])


@pytest.mark.asyncio
async def test_off_topic_question(agent):
    user_prompt = "Sicilian defense"
    result = await run_agent_test(agent, user_prompt)
    
    await assert_criteria(result, [
        "explicitly state that the question is off-topic and it only provides information about Evidently/ML evaluation",
        "return low or zero confidence (confidence <= 0.1) since the query is unrelated to the documentation",
        "use the 'search' tool",
        "don't 'get_file' tool",
        "return no references in the answer",
        "follow-up questions are about ML/LLM evaluation not about chess",
    ])


@pytest.mark.asyncio
async def test_adapts_to_user_context(agent):
    user_prompt = (
        "how do I implement LLM as a judge? I already have my input "
        "in a pandas dataframe with columns question, generated_answer, "
        "expected_answer"
    )
    result = await run_agent_test(agent, user_prompt)

    await assert_criteria(result, [
        "the answer acknowledges the user's existing pandas dataframe and references their specific column names (question, generated_answer, expected_answer)",
        "the answer does not show a generic tutorial that starts from scratch with a toy dataset, ignoring the user's existing data structure",
    ])


@pytest.mark.asyncio
async def test_uses_uv_add_not_pip(agent):
    user_prompt = "how do I install evidently?"
    result = await run_agent_test(agent, user_prompt)

    await assert_criteria(result, [
        "any package installation instructions use 'uv add' instead of 'pip install'",
    ])
