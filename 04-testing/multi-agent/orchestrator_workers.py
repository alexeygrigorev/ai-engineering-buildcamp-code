"""
Orchestrator-Workers: a central agent dynamically delegates to workers.

The orchestrator decides what to investigate next based on findings so far.
It assigns tasks to a worker agent and adapts its strategy.
Unlike parallelization, subtasks are not predefined.
"""

import asyncio
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


class TaskAssignment(BaseModel):
    task_description: str
    focus_area: str
    is_complete: bool
    guide_so_far: Optional[str] = None


orchestrator_instructions = f"""
{ONBOARDING_CONTEXT}

You are the orchestrator building an onboarding guide for {REPO}.
Based on findings so far, decide what to investigate next.

Assign specific, focused tasks to the worker. Adapt your strategy
based on what the worker discovers. When you have enough information,
set is_complete=True and provide the final guide in guide_so_far.

Think about what a newcomer needs most. Don't follow a fixed checklist -
adapt based on what you learn about the project.
""".strip()

orchestrator_agent = Agent(
    name="orchestrator",
    model="openai:gpt-4o-mini",
    instructions=orchestrator_instructions,
    output_type=TaskAssignment,
)


worker_instructions = f"""
You are a research worker exploring {REPO}.
Execute the assigned task using GitHub tools.
Return concise, specific findings. Reference actual file paths and names.
""".strip()

worker = Agent(
    name="worker",
    model="openai:gpt-4o-mini",
    instructions=worker_instructions,
    tools=tools,
)

MAX_STEPS = 6


async def main():
    question = "Help me understand scikit-learn well enough to contribute."

    print(f"Question: {question}\n")

    findings = []

    for step in range(MAX_STEPS):
        print(f"=== Orchestrator (step {step + 1}) ===\n")

        context = f"Original question: {question}\n\n"
        if findings:
            context += "Findings so far:\n" + "\n\n".join(findings)

        assignment = await orchestrator_agent.run(context)
        task = assignment.output

        if task.is_complete:
            print("Orchestrator: Guide is complete!\n")
            print(task.guide_so_far)
            break

        print(f"Task: {task.task_description}")
        print(f"Focus: {task.focus_area}\n")

        print(f"=== Worker executing ===\n")
        result = await worker.run(task.task_description)
        finding = f"[{task.focus_area}]: {result.output}"
        findings.append(finding)
        print(f"Finding: {result.output[:300]}...\n")

    else:
        print("=== Max steps reached, generating final guide ===\n")
        context = f"Findings:\n" + "\n\n".join(findings)
        final = await orchestrator_agent.run(
            f"{context}\n\nSet is_complete=True and compile the final guide."
        )
        print(final.output.guide_so_far)

if __name__ == "__main__":
    asyncio.run(main())
