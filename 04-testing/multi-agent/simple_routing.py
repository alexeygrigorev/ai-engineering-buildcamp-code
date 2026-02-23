"""
Simple Routing: classify the question, dispatch to a specialist.

The router is a classifier with no tools. It outputs a structured
QuestionCategory. Python code dispatches to the right specialist.
"""

import asyncio

from pydantic_ai import Agent

from common import REPO, ONBOARDING_CONTEXT
from github_tools import get_tools

tools = get_tools()


from typing import Literal
from pydantic import BaseModel

Category = Literal["architecture", "getting-started", "contributing", "api-usage"]

class QuestionCategory(BaseModel):
    category: Category
    reasoning: str


router_instructions = f"""
Classify the user's question about {REPO} into one of these categories:
- architecture: questions about code structure, modules, key abstractions
- getting-started: questions about setup, installation, running tests
- contributing: questions about how to contribute, good first issues, PR process
- api-usage: questions about how to use scikit-learn's API, estimators, transformers

Return the category and a brief reasoning.
""".strip()

router = Agent(
    name="router",
    model="openai:gpt-4o-mini",
    instructions=router_instructions,
    output_type=QuestionCategory,
)


architecture_instructions = f"""
{ONBOARDING_CONTEXT}

You specialize in code architecture. Focus on module structure,
key abstractions (estimator API, transformers, pipelines),
inheritance hierarchies, and how components relate to each other.
""".strip()

architecture_specialist = Agent(
    name="architecture-specialist",
    model="openai:gpt-4o-mini",
    instructions=architecture_instructions,
    tools=tools,
)


getting_started_instructions = f"""
{ONBOARDING_CONTEXT}

You specialize in getting started. Focus on installation,
development setup, running tests, and the development workflow.
""".strip()

getting_started_specialist = Agent(
    name="getting-started-specialist",
    model="openai:gpt-4o-mini",
    instructions=getting_started_instructions,
    tools=tools,
)


contributing_instructions = f"""
{ONBOARDING_CONTEXT}

You specialize in contributing. Focus on the contribution guide,
good first issues, PR review process, and maintainer expectations.
""".strip()

contributing_specialist = Agent(
    name="contributing-specialist",
    model="openai:gpt-4o-mini",
    instructions=contributing_instructions,
    tools=tools,
)


api_usage_instructions = f"""
{ONBOARDING_CONTEXT}

You specialize in API usage. Focus on the estimator API (fit/predict/transform),
common patterns, pipeline usage, and parameter conventions.
""".strip()

api_usage_specialist = Agent(
    name="api-usage-specialist",
    model="openai:gpt-4o-mini",
    instructions=api_usage_instructions,
    tools=tools,
)


specialists = {
    "architecture": architecture_specialist,
    "getting-started": getting_started_specialist,
    "contributing": contributing_specialist,
    "api-usage": api_usage_specialist,
}


async def main():
    question = "How is scikit-learn's codebase organized? What are the main modules?"
    print(f"Question: {question}\n")
    classification = await router.run(question)
    category = classification.output.category
    print(f"Category: {category}")
    print(f"Reasoning: {classification.output.reasoning}\n")
    specialist = specialists[category]
    result = await specialist.run(question)
    print(f"Specialist ({category}):\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
