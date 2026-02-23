"""
Dynamic Plan-and-Execute: plan, execute, replan after each step.

The planner creates tasks via add_task. The executor pulls tasks and
executes them with GitHub tools. The replanner reviews findings and
can add new tasks as it discovers more about the codebase.
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

github_tools = get_tools()


task_queue: list[dict] = []

def add_task(title: str, action: str, argument: str) -> str:
    """Add a research task. Actions: list_files, read_file, list_issues, search_code."""
    task_id = len(task_queue) + 1
    task_queue.append({
        "id": task_id, "title": title,
        "action": action, "argument": argument,
        "status": "pending",
    })
    return f"Added task #{task_id}: {title}"

def get_tasks() -> str:
    """Get all tasks with their current status."""
    if not task_queue:
        return "No tasks in queue."
    lines = []
    for t in task_queue:
        lines.append(f"#{t['id']} [{t['status']}] {t['title']}")
    return "\n".join(lines)


class ReplanDecision(BaseModel):
    should_replan: bool
    reasoning: str


planner_instructions = f"""
Create a research plan for onboarding newcomers to {REPO}.
Use add_task to create 4-6 research tasks.
Available actions: list_files, read_file, list_issues, search_code.
Each task should have a clear title, action, and argument.
""".strip()

planner = Agent(
    name="planner",
    model="openai:gpt-4o-mini",
    instructions=planner_instructions,
    tools=[add_task],
)


executor_instructions = f"""
{ONBOARDING_CONTEXT}

Execute the given research task using GitHub tools. Summarize findings concisely.
""".strip()

executor = Agent(
    name="executor",
    model="openai:gpt-4o-mini",
    instructions=executor_instructions,
    tools=github_tools,
)


replanner_instructions = f"""
Review the completed research and the current task queue.
Use get_tasks to see all tasks and their status.

If findings reveal something unexpected, use add_task to add new tasks.
Available actions: list_files, read_file, list_issues, search_code.

Add tasks when:
- A finding reveals something unexpected that needs investigation
- A new important area was discovered that should be explored

Keep the total remaining tasks manageable (4 or fewer).
""".strip()

replanner = Agent(
    name="replanner",
    model="openai:gpt-4o-mini",
    instructions=replanner_instructions,
    tools=[add_task, get_tasks],
    output_type=ReplanDecision,
)


synthesizer_instructions = """
Synthesize research findings into a coherent onboarding guide.
""".strip()

synthesizer = Agent(
    name="synthesizer",
    model="openai:gpt-4o-mini",
    instructions=synthesizer_instructions,
)


async def main():
    question = "Create an onboarding guide for scikit-learn contributors."

    print("=== Planning ===\n")
    await planner.run(question)
    print(f"Tasks created:\n{get_tasks()}\n")

    findings = []

    while True:
        pending = [t for t in task_queue if t["status"] == "pending"]
        if not pending:
            break

        task = pending[0]
        task["status"] = "in_progress"
        print(f"=== Executing: {task['title']} ===")

        result = await executor.run(
            f"Execute: {task['action']}({task['argument']})"
        )
        task["status"] = "done"
        findings.append(f"{task['title']}: {result.output}")
        print(f"{result.output[:200]}...\n")

        replan_context = (
            f"Completed findings:\n" + "\n".join(findings) +
            f"\n\nCurrent queue:\n{get_tasks()}"
        )
        decision = await replanner.run(replan_context)

        if decision.output.should_replan:
            print(f"Replanned: {decision.output.reasoning}")
            print(f"Queue:\n{get_tasks()}")
        else:
            print("No replanning needed.")
        print()

    print("=== Synthesis ===\n")
    all_findings = "\n\n".join(findings)
    final = await synthesizer.run(
        f"Findings:\n\n{all_findings}\n\nSynthesize into an onboarding guide."
    )
    print(final.output)

if __name__ == "__main__":
    asyncio.run(main())
