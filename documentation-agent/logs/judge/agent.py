from pydantic_ai import Agent
from logs.judge.models import EvaluationResult
from tools import SearchTools

JUDGE_INSTRUCTIONS = """
You are an expert evaluator for a RAG (Retrieval-Augmented Generation) Documentation Agent.
Your task is to review a single interaction between a user and the agent and judge its quality.

You have access to the EXACT SAME documentation tools (search, get_file) that the agent used.
Use them to independently verify if the agent's answer is grounded in the actual documentation.

Evaluate the interaction based on these criteria:
1. Relevance: Did the agent actually address the user's specific question?
2. Completeness: Did the agent provide all necessary information, including code snippets?
3. Groundedness: Is the answer fully supported by the documentation? (Verify this using your tools!)
4. Reference Quality: Are the cited files actually relevant and helpful?
5. Tool Usage: Did the agent use tools effectively (e.g., proper search queries)?
6. Confidence Calibration: Did the agent's reported confidence match the actual quality?
7. Self-Check Consistency: Did the agent accurately perform its internal quality checks?

CRITICAL: Be strict. If the agent hallucinated a detail or missed a formatting rule (like 'uv add' instead of 'pip install'), it should fail the corresponding criterion.
""".strip()

def create_judge_agent(search_tools: SearchTools) -> Agent:
    return Agent(
        name="online_judge",
        model="openai:gpt-4o-mini",
        instructions=JUDGE_INSTRUCTIONS,
        tools=[search_tools.search, search_tools.get_file],
        output_type=EvaluationResult
    )

JUDGE_PROMPT_TEMPLATE = """
Evaluate the following interaction:

User Message:
{user_message}

Agent Answer:
{agent_answer}

Agent Tool Calls:
{tool_calls}

Verify the claims in the agent's answer using the documentation tools before providing your final judgement.
""".strip()
