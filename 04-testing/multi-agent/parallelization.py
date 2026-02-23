"""
Parallelization: multiple analysts run simultaneously, results are aggregated.

Four agents analyze different aspects of the repo in parallel via
asyncio.gather(). An aggregator combines their findings.
Includes a sequential comparison for timing.
"""

import asyncio
import time

from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


code_analyst_instructions = f"""
Analyze the code structure of {REPO}.
Focus on: main packages, key abstractions, inheritance patterns, module organization.
Use the GitHub tools to explore the codebase. Be concise.
""".strip()

code_analyst = Agent(
    name="code-analyst",
    model="openai:gpt-4o-mini",
    instructions=code_analyst_instructions,
    tools=tools,
)


docs_analyst_instructions = f"""
Analyze the documentation of {REPO}.
Focus on: README quality, developer guides, docstring conventions, examples.
Use the GitHub tools to read documentation files. Be concise.
""".strip()

docs_analyst = Agent(
    name="docs-analyst",
    model="openai:gpt-4o-mini",
    instructions=docs_analyst_instructions,
    tools=tools,
)


community_analyst_instructions = f"""
Analyze the community aspects of {REPO}.
Focus on: issue activity, good first issues, PR review patterns, maintainer activity.
Use the GitHub tools to explore issues and activity. Be concise.
""".strip()

community_analyst = Agent(
    name="community-analyst",
    model="openai:gpt-4o-mini",
    instructions=community_analyst_instructions,
    tools=tools,
)


testing_analyst_instructions = f"""
Analyze the testing setup of {REPO}.
Focus on: test framework, CI configuration, how to run tests locally, test organization.
Use the GitHub tools to explore test infrastructure. Be concise.
""".strip()

testing_analyst = Agent(
    name="testing-analyst",
    model="openai:gpt-4o-mini",
    instructions=testing_analyst_instructions,
    tools=tools,
)


aggregator_instructions = """
Combine analysis results from multiple specialists into a
coherent onboarding guide. Organize by topic, highlight key findings,
and resolve any contradictions between analysts.
""".strip()

aggregator = Agent(
    name="aggregator",
    model="openai:gpt-4o-mini",
    instructions=aggregator_instructions,
)

analysts = [
    ("Code Structure", code_analyst),
    ("Documentation", docs_analyst),
    ("Community", community_analyst),
    ("Testing", testing_analyst),
]


async def run_analyst(name, agent):
    result = await agent.run(f"Analyze the {name.lower()} aspects of {REPO}.")
    return name, result.output

async def run_parallel():
    print("=== Parallel Execution ===\n")
    start = time.time()

    tasks = [run_analyst(name, agent) for name, agent in analysts]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print(f"Parallel execution: {elapsed:.1f}s\n")

    for name, output in results:
        print(f"--- {name} ---")
        print(f"{output[:200]}...\n")

    return results, elapsed


async def run_sequential():
    print("=== Sequential Execution ===\n")
    start = time.time()

    results = []
    for name, agent in analysts:
        name, output = await run_analyst(name, agent)
        results.append((name, output))

    elapsed = time.time() - start
    print(f"Sequential execution: {elapsed:.1f}s\n")

    return results, elapsed


async def main():
    parallel_results, parallel_time = await run_parallel()

    sequential_results, sequential_time = await run_sequential()

    print(f"=== Timing Comparison ===")
    print(f"Parallel:   {parallel_time:.1f}s")
    print(f"Sequential: {sequential_time:.1f}s")
    print(f"Speedup:    {sequential_time / parallel_time:.1f}x\n")

    print("=== Aggregating Results ===\n")
    findings = "\n\n".join(
        f"## {name}\n{output}" for name, output in parallel_results
    )
    final = await aggregator.run(
        f"Combine these analysis results into an onboarding guide:\n\n{findings}"
    )
    print(final.output)

if __name__ == "__main__":
    asyncio.run(main())
