from pydantic_ai import Agent, RunUsage

from doc_agent import (
    DocumentationAgentConfig,
    create_agent,
    run_agent,
    run_agent_stream,
    DEFAULT_INSTRUCTIONS
)

from tools import create_documentation_tools_cached

import dotenv
dotenv.load_dotenv()


def print_messages(messages):
    for m in messages:
        print(m.kind)
        for p in m.parts:
            part_kind = p.part_kind
            if part_kind == 'user-prompt':
                print('  USER:', p.content)
            if part_kind == 'tool-call':
                print('  TOOL CALL:', p.tool_name, p.args)
            if part_kind == 'tool-return':
                print('  TOOL RETURN:', p.tool_name)
            if part_kind == 'text':
                print('  ASSISTANT:', p.content)
        print()


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


async def run_agent_question_blocking():
    user_prompt = "LLM as a judge"
    print(f"Running agent with question: {user_prompt}...")

    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()

    agent = create_agent(agent_config, tools)

    result = await run_agent(agent, user_prompt)
    print_messages(result.new_messages())
    print(result.output)


async def run_agent_question():
    user_prompt = "LLM as a judge"
    print(f"Running agent with question: {user_prompt}...")

    tools = create_documentation_tools_cached()
    agent_config = DocumentationAgentConfig()

    agent = create_agent(agent_config, tools)

    result = await run_agent_stream(agent, user_prompt)
    print_messages(result.new_messages())


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_agent_question())
