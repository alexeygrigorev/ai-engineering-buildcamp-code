"""
Agents as Tools: the coordinator delegates context-heavy subtasks to subagents.

Each tool function wraps a subagent that processes large amounts of data
(file listings, issue lists, code search results) and returns a concise summary.
The coordinator's context stays clean.
"""

import asyncio

from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


code_explorer_instructions = f"""
You explore the {REPO} codebase structure.
Use the GitHub tools to list directories, read key files, and understand
the module organization. Return a concise summary of what you find.
""".strip()

code_explorer = Agent(
    name="code-explorer",
    model="openai:gpt-4o-mini",
    instructions=code_explorer_instructions,
    tools=tools,
)


docs_reader_instructions = f"""
You read documentation files in {REPO}.
Use the GitHub tools to find and read README, CONTRIBUTING, and other docs.
Return a concise summary of the key information.
""".strip()

docs_reader = Agent(
    name="docs-reader",
    model="openai:gpt-4o-mini",
    instructions=docs_reader_instructions,
    tools=tools,
)


issue_analyst_instructions = f"""
You analyze issues in {REPO}.
Use the GitHub tools to list and search issues. Identify patterns,
common topics, and good first issues. Return a concise summary.
""".strip()

issue_analyst = Agent(
    name="issue-analyst",
    model="openai:gpt-4o-mini",
    instructions=issue_analyst_instructions,
    tools=tools,
)


coordinator_instructions = f"""
{ONBOARDING_CONTEXT}

You coordinate the onboarding guide creation by delegating to specialized tools.
Each tool handles a context-heavy subtask and returns a summary.
Use the tools to gather information, then synthesize into a guide.
""".strip()

coordinator = Agent(
    name="coordinator",
    model="openai:gpt-4o-mini",
    instructions=coordinator_instructions,
)


@coordinator.tool_plain
async def explore_code(focus_area: str) -> str:
    """Explore the codebase structure, focusing on a specific area.

    Args:
        focus_area: What to explore (e.g., "top-level modules", "sklearn/ensemble").
    """
    result = await code_explorer.run(
        f"Explore the {focus_area} in {REPO}. List key files and describe the structure."
    )
    return result.output


@coordinator.tool_plain
async def read_docs(doc_type: str) -> str:
    """Read and summarize documentation files.

    Args:
        doc_type: Type of docs to read (e.g., "README", "CONTRIBUTING", "installation").
    """
    result = await docs_reader.run(
        f"Find and read the {doc_type} documentation in {REPO}. Summarize the key points."
    )
    return result.output


@coordinator.tool_plain
async def analyze_issues(focus: str) -> str:
    """Analyze repository issues for patterns and opportunities.

    Args:
        focus: What to analyze (e.g., "good first issues", "recent bugs", "feature requests").
    """
    result = await issue_analyst.run(
        f"Analyze {focus} in {REPO}. Identify patterns and key findings."
    )
    return result.output


async def main():
    question = "Help me onboard onto scikit-learn. What should I know?"
    print(f"Question: {question}\n")
    result = await coordinator.run(question)
    print(f"Onboarding Guide:\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
