"""
Workflow vs Orchestrated: the two fundamental approaches to multi-agent systems.

Workflow: a fixed for-loop runs agents in sequence. The code controls the flow.
Orchestrated: a coordinator agent decides which tools to call and in what order.
"""

import asyncio

from pydantic_ai import Agent

from common import REPO
from github_tools import get_tools

tools = get_tools()


explorer_instructions = f"""
List the top-level modules in {REPO}.
Use list_files to look at the root directory.
Return a short bullet list of the main directories.
""".strip()

explorer = Agent(
    name="explorer",
    model="openai:gpt-4o-mini",
    instructions=explorer_instructions,
    tools=tools,
)


summarizer_instructions = """
Given a list of modules, write a 2-3 sentence
summary of what this project is about based on the module names.
""".strip()

summarizer = Agent(
    name="summarizer",
    model="openai:gpt-4o-mini",
    instructions=summarizer_instructions,
)


async def workflow():
    # Step 1: explorer lists modules
    result1 = await explorer.run(f"List the top-level modules in {REPO}.")
    modules = result1.output
    print(f"=== Explorer ===\n{modules}\n")

    # Step 2: summarizer interprets the list
    result2 = await summarizer.run(f"Modules:\n{modules}")
    print(f"=== Summarizer ===\n{result2.output}\n")


coordinator_instructions = f"""
You are exploring {REPO} to help a newcomer.
Use the available tools to list modules and then summarize the project.
Produce a short summary of what the project is about.
""".strip()

coordinator = Agent(
    name="coordinator",
    model="openai:gpt-4o-mini",
    instructions=coordinator_instructions,
)


@coordinator.tool_plain
async def list_modules() -> str:
    """List top-level modules in the repository."""
    result = await explorer.run(f"List the top-level modules in {REPO}.")
    return result.output

@coordinator.tool_plain
async def summarize_modules(module_list: str) -> str:
    """Summarize what the project does based on its modules.

    Args:
        module_list: A list of module names to summarize.
    """
    result = await summarizer.run(f"Modules:\n{module_list}")
    return result.output


async def orchestrated():
    result = await coordinator.run("What is this project about?")
    print(f"=== Coordinator ===\n{result.output}\n")


if __name__ == "__main__":
    print("--- Workflow ---\n")
    asyncio.run(workflow())

    print("--- Orchestrated ---\n")
    asyncio.run(orchestrated())
