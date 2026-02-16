from pydantic_ai import Agent, RunUsage

from doc_agent import DocumentationAgentConfig, create_agent, run_agent
from tools import create_documentation_tools_cached

import dotenv
dotenv.load_dotenv()


async def run_qna(agent: Agent):
    messages = []
    usage = RunUsage()

    while True:
        user_prompt = input('You:')
        if user_prompt.lower().strip() == 'stop':
            break

        user_prompt = "What is this repository about?"
        result = await run_agent(agent, user_prompt, messages)

        usage = usage + result.usage()
        messages.extend(result.new_messages())


async def run_agent_question():
    user_prompt = "LLM as a judge"
    print(f"Running agent with question: {user_prompt}...")
    
    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()

    agent = create_agent(agent_config, tools)

    result = await run_agent(agent, user_prompt)
    print(result.output)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_agent_question())
