"""
Human-in-the-Loop: the agent pauses for human review at checkpoints.

The writer generates a draft. The script pauses with input() for
human feedback. The human can approve or request changes.
"""

import asyncio

from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


writer_instructions = f"""
{ONBOARDING_CONTEXT}

Write or revise onboarding guide sections about {REPO}.
Use the GitHub tools to gather accurate information.
Incorporate human feedback precisely - address every point raised.
""".strip()

writer = Agent(
    name="writer",
    model="openai:gpt-4o-mini",
    instructions=writer_instructions,
    tools=tools,
)

MAX_REVISIONS = 5


async def main():
    topic = f"getting started contributing to {REPO}"

    print(f"=== Generating draft about: {topic} ===\n")
    result = await writer.run(f"Write a guide section about {topic}.")
    content = result.output

    for revision in range(MAX_REVISIONS):
        print(f"\n{'=' * 60}")
        print(f"Draft (revision {revision}):")
        print(f"{'=' * 60}\n")
        print(content)
        print(f"\n{'=' * 60}")

        feedback = input(
            "\nYour feedback (press Enter to approve, or type feedback): "
        ).strip()

        if not feedback or feedback.lower() in ("ok", "approve", "lgtm", "yes"):
            print("\n=== Approved! ===")
            break

        print(f"\n=== Revising based on feedback ===\n")
        result = await writer.run(
            f"Revise this draft based on human feedback:\n\n"
            f"Current draft:\n{content}\n\n"
            f"Human feedback:\n{feedback}"
        )
        content = result.output

    print(f"\nFinal approved content:\n{content}")


if __name__ == "__main__":
    asyncio.run(main())
