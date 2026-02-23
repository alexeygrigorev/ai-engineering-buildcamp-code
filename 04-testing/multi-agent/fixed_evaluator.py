"""
Fixed Evaluator-Optimizer: generate, evaluate, loop until good enough.

A writer generates a guide section. An evaluator with GitHub tools
checks facts against the actual repo. The loop is hardcoded:
write → evaluate → rewrite until score >= 4 or max iterations.
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


class Evaluation(BaseModel):
    score: int  # 1-5
    is_acceptable: bool  # True if score >= 4
    feedback: str  # specific issues found
    factual_errors: list[str]  # incorrect claims about the repo


writer_instructions = f"""
{ONBOARDING_CONTEXT}

Write a section about the module structure of {REPO}.
Cover the main packages, what each one does, and how they relate.
Use the GitHub tools to explore the actual codebase.
""".strip()

writer = Agent(
    name="writer",
    model="openai:gpt-4o-mini",
    instructions=writer_instructions,
    tools=tools,
)


evaluator_instructions = f"""
Evaluate a guide section about {REPO} for accuracy and completeness.

Use the GitHub tools to verify claims:
- Do the mentioned file paths exist?
- Are the module descriptions accurate?
- Is anything important missing?

Score from 1-5:
- 1-2: Major factual errors or missing critical information
- 3: Mostly correct but has gaps or minor errors
- 4-5: Accurate and comprehensive

Be specific in your feedback - reference actual file paths and class names.
""".strip()

evaluator = Agent(
    name="evaluator",
    model="openai:gpt-4o-mini",
    instructions=evaluator_instructions,
    output_type=Evaluation,
    tools=tools,
)

MAX_ITERATIONS = 3


async def main():
    topic = f"the module structure of {REPO}"

    print(f"=== Writing about: {topic} ===\n")

    draft = await writer.run(f"Write a guide section about {topic}.")
    content = draft.output
    print(f"Draft:\n{content[:300]}...\n")

    for i in range(MAX_ITERATIONS):
        print(f"=== Evaluation (iteration {i + 1}) ===\n")

        eval_result = await evaluator.run(
            f"Evaluate this guide section:\n\n{content}"
        )
        evaluation = eval_result.output

        print(f"Score: {evaluation.score}/5")
        print(f"Acceptable: {evaluation.is_acceptable}")
        print(f"Feedback: {evaluation.feedback}")
        if evaluation.factual_errors:
            print(f"Factual errors: {evaluation.factual_errors}")
        print()

        if evaluation.is_acceptable:
            print("=== Accepted! ===\n")
            break

        print("=== Rewriting ===\n")
        draft = await writer.run(
            f"Rewrite this guide section based on the evaluator's feedback:\n\n"
            f"Current content:\n{content}\n\n"
            f"Feedback:\n{evaluation.feedback}\n\n"
            f"Factual errors to fix:\n" + "\n".join(evaluation.factual_errors)
        )
        content = draft.output
        print(f"Revised:\n{content[:300]}...\n")

    print(f"Final content:\n{content}")


if __name__ == "__main__":
    asyncio.run(main())
