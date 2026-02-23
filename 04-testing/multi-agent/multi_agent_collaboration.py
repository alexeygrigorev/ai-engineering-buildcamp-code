"""
Multi-Agent Collaboration: peers collaborate in a shared conversation.

Three agents (architect, developer, reviewer) take turns contributing
their perspective. Each sees the full conversation history.
A compiler reorganizes the discussion by topic.
"""

import asyncio

from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


architect_instructions = f"""
You are the architect in a team creating an onboarding guide for {REPO}.

Your expertise: code structure, design patterns, module organization,
key abstractions (estimator API, transformers, pipelines).

You see the full conversation. Build on what others said.
If you disagree, explain why. Keep contributions focused and concise.
Use the GitHub tools to verify your claims.
""".strip()

architect = Agent(
    name="architect",
    model="openai:gpt-4o-mini",
    instructions=architect_instructions,
    tools=tools,
)


developer_instructions = f"""
You are the developer in a team creating an onboarding guide for {REPO}.

Your expertise: practical development workflow - setup, testing,
debugging, common pitfalls, tooling, CI/CD.

You see the full conversation. Build on what others said.
Focus on what a new contributor actually needs to DO.
Use the GitHub tools to find practical information.
""".strip()

developer = Agent(
    name="developer",
    model="openai:gpt-4o-mini",
    instructions=developer_instructions,
    tools=tools,
)


reviewer_instructions = f"""
You are the reviewer in a team creating an onboarding guide for {REPO}.

Your expertise: contribution process, code review standards,
good first issues, community norms, documentation quality.

You see the full conversation. Build on what others said.
Push back if something is too complex for newcomers.
Use the GitHub tools to check issues and contribution guidelines.
""".strip()

reviewer = Agent(
    name="reviewer",
    model="openai:gpt-4o-mini",
    instructions=reviewer_instructions,
    tools=tools,
)


compiler_instructions = """
Compile a team discussion into a structured onboarding guide.
Reorganize by topic (not by speaker). Resolve disagreements.
Keep the practical, newcomer-friendly tone.
""".strip()

compiler = Agent(
    name="compiler",
    model="openai:gpt-4o-mini",
    instructions=compiler_instructions,
)

agents = [
    ("Architect", architect),
    ("Developer", developer),
    ("Reviewer", reviewer),
]

NUM_ROUNDS = 2


async def main():
    topic = f"Creating an onboarding guide for {REPO}"

    conversation = [f"Topic: {topic}\n\nLet's each contribute our perspective."]

    for round_num in range(NUM_ROUNDS):
        print(f"\n{'=' * 60}")
        print(f"Round {round_num + 1}")
        print(f"{'=' * 60}\n")

        for agent_name, agent in agents:
            print(f"--- {agent_name} ---\n")

            prompt = "\n\n".join(conversation)
            if round_num > 0:
                prompt += f"\n\nThis is round {round_num + 1}. Build on the previous discussion. Add new insights or push back on points you disagree with."

            result = await agent.run(prompt)
            contribution = f"[{agent_name}]: {result.output}"
            conversation.append(contribution)
            print(f"{result.output[:300]}...\n")

    print(f"\n{'=' * 60}")
    print("Compiling final guide")
    print(f"{'=' * 60}\n")

    full_discussion = "\n\n".join(conversation)
    final = await compiler.run(
        f"Compile this team discussion into an onboarding guide:\n\n{full_discussion}"
    )
    print(final.output)

if __name__ == "__main__":
    asyncio.run(main())
