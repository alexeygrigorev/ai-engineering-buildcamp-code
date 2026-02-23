from pydantic import BaseModel, Field
from pydantic_ai import Agent 

from tests.utils import collect_tools, get_model_name
from tests.cost_tracker import capture_usage


judge_instructions = f"""
You are an expert judge evaluating the performance of an
AI agent.
""".strip()

class JudgeCriterion(BaseModel):
    """
    Evaluation of a single test requirement or behavioral rule.
    """
    criterion_description: str = Field(
        description="The specific requirement or rule that the agent is being evaluated against."
    )
    passed: bool = Field(
        description="Indicates whether the agent's response and actions successfully satisfied this requirement."
    )
    judgement: str = Field(
        description="A clear explanation of why the agent passed or failed, referencing specific evidence from the agent's output or tool calls."
    )


class JudgeFeedback(BaseModel):
    """
    The complete evaluation report from the judge agent, summarizing performance across all criteria.
    """
    criteria: list[JudgeCriterion] = Field(
        description="A collection of individual evaluations for each performance requirement provided in the test."
    )
    feedback: str = Field(
        description="A holistic summary of the agent's performance, providing overall context and identifying key strengths or failures."
    )



def create_judge_agent():
    agent = Agent(
        name="judge",
        model="openai:gpt-4o-mini",
        instructions=judge_instructions,
        output_type=JudgeFeedback
    )
    return agent


judge_user_prompt_template = """
Evaluate the agent's performance based on the following criteria:
<CRITERIA>
{criteria}
</CRITERIA>

The agent's final output was:
<AGENT_OUTPUT>
{output}
</AGENT_OUTPUT>

Tool calls:
<TOOL_CALLS>
{tool_calls}
</TOOL_CALLS>
""".strip()


async def assert_criteria(result, criteria):
    messages = result.new_messages()
    tool_calls = collect_tools(messages)
    output = str(result.output)

    judge_agent = create_judge_agent()
    judge_user_prompt = judge_user_prompt_template.format(
        criteria='\n'.join(criteria),
        output=output,
        tool_calls='\n'.join([str(tc) for tc in tool_calls])
    )

    print(judge_user_prompt)

    judge_result = await judge_agent.run(judge_user_prompt)

    model = get_model_name(judge_agent)
    capture_usage(model, judge_result)

    print('judge feedback:')
    print(judge_result.output.feedback)

    for criterion in judge_result.output.criteria:
        print(f"{criterion.criterion_description}: {criterion.judgement}")
        assert criterion.passed, f"{criterion.criterion_description}: {criterion.judgement}"

