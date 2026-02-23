"""
Static Plan-and-Execute: planner creates a fixed plan, executor follows it.

The planner generates a multi-step research plan upfront.
The executor runs each step in order using GitHub tools.
No replanning - the plan is fixed once created.
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


class PlanStep(BaseModel):
    step_number: int
    action: str  # list_files, read_file, list_issues, search_code
    argument: str  # the argument to pass to the action
    reason: str  # why this step is needed

class ResearchPlan(BaseModel):
    goal: str
    steps: list[PlanStep]


planner_instructions = f"""
Create a research plan for exploring {REPO}.
The available actions are:
- list_files(path): list files at a directory path
- read_file(path): read a file's contents
- list_issues(label, limit): list open issues, optionally by label
- search_code(query): search for code patterns

Create a plan with 4-6 concrete steps. Each step should have a specific
action, argument, and reason. The goal is to gather information for
an onboarding guide.
""".strip()

planner = Agent(
    name="planner",
    model="openai:gpt-4o-mini",
    instructions=planner_instructions,
    output_type=ResearchPlan,
)


executor_instructions = f"""
{ONBOARDING_CONTEXT}

You are executing a research plan. For each step, use the appropriate
GitHub tool and summarize what you found. Be concise but specific.
""".strip()

executor = Agent(
    name="executor",
    model="openai:gpt-4o-mini",
    instructions=executor_instructions,
    tools=tools,
)


synthesizer_instructions = """
Take the research findings and synthesize them into
a coherent onboarding guide. Organize by topic, not by step order.
""".strip()

synthesizer = Agent(
    name="synthesizer",
    model="openai:gpt-4o-mini",
    instructions=synthesizer_instructions,
)


async def main():
    question = "Create an onboarding guide for scikit-learn contributors."

    print("=== Planning Phase ===\n")
    plan_result = await planner.run(question)
    plan = plan_result.output

    print(f"Goal: {plan.goal}\n")
    for step in plan.steps:
        print(f"  Step {step.step_number}: {step.action}({step.argument})")
        print(f"    Reason: {step.reason}")
    print()

    print("=== Execution Phase ===\n")
    findings = []

    for step in plan.steps:
        print(f"--- Executing Step {step.step_number}: {step.action}({step.argument}) ---")
        result = await executor.run(
            f"Execute this research step:\n"
            f"Action: {step.action}\n"
            f"Argument: {step.argument}\n"
            f"Reason: {step.reason}\n\n"
            f"Use the appropriate GitHub tool and summarize your findings."
        )
        findings.append(f"Step {step.step_number} ({step.action}): {result.output}")
        print(f"Finding: {result.output[:200]}...\n")

    print("=== Synthesis Phase ===\n")
    all_findings = "\n\n".join(findings)
    final = await synthesizer.run(
        f"Research findings:\n\n{all_findings}\n\n"
        f"Synthesize these into a coherent onboarding guide."
    )
    print(final.output)

if __name__ == "__main__":
    asyncio.run(main())
