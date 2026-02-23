"""
Prompt Chaining: a fixed sequence of LLM calls, each processing the previous output.

Four steps:
1. README analyst - reads README, produces project overview
2. Module mapper - takes overview + directory listing, maps key modules
3. Setup writer - takes module map, writes getting-started section
4. Guide compiler - takes everything, formats the final onboarding guide
"""

import asyncio

from pydantic_ai import Agent

from common import REPO
from github_tools import get_tools

tools = get_tools()


readme_analyst_instructions = f"""
Read the README of {REPO} and produce a concise project overview.
Cover: what the project does, who it's for, key features, and the tech stack.
Use the GitHub tools to read the README file.
""".strip()

readme_analyst = Agent(
    name="readme-analyst",
    model="openai:gpt-4o-mini",
    instructions=readme_analyst_instructions,
    tools=tools,
)


module_mapper_instructions = f"""
Given a project overview and directory listing, identify the key modules
in {REPO} and how they relate to each other.
Focus on: main packages, key abstractions, and the dependency structure.
Use the GitHub tools to explore the directory structure.
""".strip()

module_mapper = Agent(
    name="module-mapper",
    model="openai:gpt-4o-mini",
    instructions=module_mapper_instructions,
    tools=tools,
)


setup_writer_instructions = f"""
Given a module map, write a "Getting Started" section for newcomers to {REPO}.
Cover: installation, dev setup, running tests, and the basic contribution workflow.
Use the GitHub tools to read relevant files like CONTRIBUTING.md.
""".strip()

setup_writer = Agent(
    name="setup-writer",
    model="openai:gpt-4o-mini",
    instructions=setup_writer_instructions,
    tools=tools,
)


guide_compiler_instructions = """
Compile all the sections into a well-structured onboarding guide.
Organize the content logically. Don't repeat information.
Format it as a clean, readable document with clear sections.
""".strip()

guide_compiler = Agent(
    name="guide-compiler",
    model="openai:gpt-4o-mini",
    instructions=guide_compiler_instructions,
)


async def main():
    print("=== Step 1: README Analysis ===\n")
    step1 = await readme_analyst.run(
        f"Read the README of {REPO} and create a project overview."
    )
    overview = step1.output
    print(f"{overview[:300]}...\n")

    print("=== Step 2: Module Mapping ===\n")
    step2 = await module_mapper.run(
        f"Project overview:\n{overview}\n\n"
        f"Now explore the directory structure of {REPO} and map the key modules."
    )
    module_map = step2.output
    print(f"{module_map[:300]}...\n")

    print("=== Step 3: Setup Guide ===\n")
    step3 = await setup_writer.run(
        f"Module map:\n{module_map}\n\n"
        f"Write a Getting Started section for newcomers."
    )
    setup_guide = step3.output
    print(f"{setup_guide[:300]}...\n")

    print("=== Step 4: Final Guide ===\n")
    step4 = await guide_compiler.run(
        f"Compile this into a final onboarding guide:\n\n"
        f"## Project Overview\n{overview}\n\n"
        f"## Module Map\n{module_map}\n\n"
        f"## Getting Started\n{setup_guide}"
    )
    print(step4.output)

if __name__ == "__main__":
    asyncio.run(main())
