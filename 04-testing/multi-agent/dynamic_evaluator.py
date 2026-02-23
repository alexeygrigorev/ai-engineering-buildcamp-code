"""
Dynamic Evaluator-Optimizer: orchestrator decides when and how to evaluate.

Unlike the fixed loop, an orchestrator agent decides the next action:
accept the draft, rewrite it, fact-check specific claims, or expand sections.
"""

import asyncio
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


class FactCheckResult(BaseModel):
    claims_checked: list[str]
    errors_found: list[str]
    verified: list[str]

class NextAction(BaseModel):
    action: str  # accept, rewrite, fact_check, expand
    reasoning: str
    instructions: Optional[str] = None  # specific instructions for the action


writer_instructions = f"""
{ONBOARDING_CONTEXT}

Write or revise onboarding guide sections about {REPO}.
Use the GitHub tools to gather accurate information.
""".strip()

writer = Agent(
    name="writer",
    model="openai:gpt-4o-mini",
    instructions=writer_instructions,
    tools=tools,
)


fact_checker_instructions = f"""
Fact-check claims about {REPO} against the actual repository.
Use the GitHub tools to verify file paths, class names, and descriptions.
Be thorough - check specific claims, not just general accuracy.
""".strip()

fact_checker = Agent(
    name="fact-checker",
    model="openai:gpt-4o-mini",
    instructions=fact_checker_instructions,
    output_type=FactCheckResult,
    tools=tools,
)


orchestrator_instructions = f"""
You manage the quality of an onboarding guide for {REPO}.

Review the current draft and any feedback, then decide the next action:
- accept: the draft is good enough to publish
- rewrite: the draft needs revision (provide specific instructions)
- fact_check: specific claims need verification (list what to check)
- expand: a section needs more detail (specify which section)

Be strategic - don't just loop. If fact-checking found errors, rewrite.
If the draft is thin on a topic, expand. If it's solid, accept.
""".strip()

orchestrator = Agent(
    name="orchestrator",
    model="openai:gpt-4o-mini",
    instructions=orchestrator_instructions,
    output_type=NextAction,
)

MAX_STEPS = 5


async def main():
    topic = f"the estimator API pattern in {REPO}"

    print(f"=== Initial Draft: {topic} ===\n")
    draft = await writer.run(f"Write about {topic}.")
    content = draft.output
    print(f"Draft:\n{content[:300]}...\n")

    history = []

    for step in range(MAX_STEPS):
        print(f"=== Orchestrator Decision (step {step + 1}) ===\n")

        context = (
            f"Current draft:\n{content}\n\n"
            f"History:\n" + "\n".join(history) if history else f"Current draft:\n{content}"
        )
        decision = await orchestrator.run(context)
        action = decision.output

        print(f"Action: {action.action}")
        print(f"Reasoning: {action.reasoning}")
        if action.instructions:
            print(f"Instructions: {action.instructions}")
        print()

        if action.action == "accept":
            print("=== Accepted! ===\n")
            break

        elif action.action == "rewrite":
            result = await writer.run(
                f"Revise this draft:\n\n{content}\n\n"
                f"Instructions: {action.instructions}"
            )
            content = result.output
            history.append(f"Rewrote draft: {action.instructions}")
            print(f"Revised:\n{content[:200]}...\n")

        elif action.action == "fact_check":
            result = await fact_checker.run(
                f"Fact-check these claims in the draft:\n\n{content}\n\n"
                f"Focus on: {action.instructions}"
            )
            check = result.output
            history.append(
                f"Fact-check: verified {len(check.verified)}, "
                f"errors {len(check.errors_found)}: {check.errors_found}"
            )
            print(f"Verified: {check.verified}")
            print(f"Errors: {check.errors_found}\n")

        elif action.action == "expand":
            result = await writer.run(
                f"Expand this section of the draft:\n\n{content}\n\n"
                f"What to expand: {action.instructions}"
            )
            content = result.output
            history.append(f"Expanded: {action.instructions}")
            print(f"Expanded:\n{content[:200]}...\n")

    print(f"Final content:\n{content}")


if __name__ == "__main__":
    asyncio.run(main())
