"""
Handoffs: the active agent transfers control to another agent.

Each agent returns a HandoffDecision. If handoff_to is set,
control passes to that agent with the context summary.
The conversation flows between agents until one completes without a handoff.
"""

import asyncio

from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


from typing import Optional
from pydantic import BaseModel

class HandoffDecision(BaseModel):
    response: str
    handoff_to: Optional[str] = None
    context_summary: Optional[str] = None

AVAILABLE_AGENTS = ["triage", "architecture", "setup", "contributing"]

HANDOFF_INSTRUCTIONS = f"""
If the user's question is better handled by a different specialist, set handoff_to
to one of: {', '.join(AVAILABLE_AGENTS)}.
Include a context_summary so the next agent knows what was discussed.
If you can fully answer the question, leave handoff_to as null.
""".strip()


triage_instructions = f"""
You are the triage agent for {REPO} onboarding questions.
Determine which specialist should handle the question:
- architecture: code structure, modules, abstractions
- setup: installation, dev environment, running tests
- contributing: PRs, issues, contribution workflow

Briefly acknowledge the question and hand off to the right specialist.
{HANDOFF_INSTRUCTIONS}
""".strip()

triage_agent = Agent(
    name="triage",
    model="openai:gpt-4o-mini",
    instructions=triage_instructions,
    output_type=HandoffDecision,
    tools=tools,
)


architecture_instructions = f"""
{ONBOARDING_CONTEXT}

You specialize in code architecture for {REPO}. Focus on module structure,
key abstractions, and how components relate.
{HANDOFF_INSTRUCTIONS}
""".strip()

architecture_agent = Agent(
    name="architecture",
    model="openai:gpt-4o-mini",
    instructions=architecture_instructions,
    output_type=HandoffDecision,
    tools=tools,
)


setup_instructions = f"""
{ONBOARDING_CONTEXT}

You specialize in development setup for {REPO}. Focus on installation,
environment configuration, and running tests.
{HANDOFF_INSTRUCTIONS}
""".strip()

setup_agent = Agent(
    name="setup",
    model="openai:gpt-4o-mini",
    instructions=setup_instructions,
    output_type=HandoffDecision,
    tools=tools,
)


contributing_instructions = f"""
{ONBOARDING_CONTEXT}

You specialize in contributing to {REPO}. Focus on the contribution workflow,
good first issues, and PR process.
{HANDOFF_INSTRUCTIONS}
""".strip()

contributing_agent = Agent(
    name="contributing",
    model="openai:gpt-4o-mini",
    instructions=contributing_instructions,
    output_type=HandoffDecision,
    tools=tools,
)


agents = {
    "triage": triage_agent,
    "architecture": architecture_agent,
    "setup": setup_agent,
    "contributing": contributing_agent,
}


async def main():
    question = "How do I set up the dev environment and run tests in scikit-learn?"
    print(f"Question: {question}\n")

    current_agent_name = "triage"
    context = question
    max_handoffs = 3

    for i in range(max_handoffs + 1):
        agent = agents[current_agent_name]
        print(f"--- Agent: {current_agent_name} ---")

        result = await agent.run(context)
        decision = result.output
        print(f"Response: {decision.response[:200]}...")

        if decision.handoff_to and decision.handoff_to != current_agent_name:
            print(f"Handing off to: {decision.handoff_to}")
            print(f"Context: {decision.context_summary}\n")
            current_agent_name = decision.handoff_to
            context = f"Previous context: {decision.context_summary}\n\nOriginal question: {question}"
        else:
            print(f"\nFinal response from {current_agent_name}:")
            print(decision.response)
            break


if __name__ == "__main__":
    asyncio.run(main())
