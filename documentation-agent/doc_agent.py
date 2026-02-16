from dataclasses import dataclass

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import FunctionToolCallEvent

from tools import SearchTools


DEFAULT_INSTRUCTIONS = """
You're a documentation assistant.

Answer the user question using only the documentation knowledge base.

Make 3 iterations:

1) First iteration:
   - Perform one search using the search tool to identify potentially relevant documents.
   - Explain (in 2-3 sentences) why this search query is appropriate for the user question.

2) Second iteration:
   - Analyze the results from the previous search.
   - Based on the filenames or documents returned, perform:
       - Up to 2 additional search queries to refine or expand coverage, and
       - One or more get_file calls to retrieve the full content of the most relevant documents.
   - For each search or get_file call, explain (in 2-3 sentences) why this action is necessary and how it helps answer the question.

3) Third iteration:
   - Analyze the retrieved document contents from get_file.
   - Synthesize the information from these documents into a final answer to the user.

IMPORTANT:
- At every step, explicitly explain your reasoning for each search query or file retrieval.
- Use only facts found in the documentation knowledge base.
- Do not introduce outside knowledge or assumptions.
- If the answer cannot be found in the retrieved documents, clearly inform the user.

Additional notes:
- The knowledge base is entirely about Evidently, so you do not need to include the word "evidently" in search queries.
- Prefer retrieving and analyzing full documents (via get_file) before producing the final answer.
""".strip()



@dataclass
class DocumentationAgentConfig:
    model = 'openai:gpt-4o-mini'
    name = 'search'
    instructions = DEFAULT_INSTRUCTIONS


def create_agent(
        config: DocumentationAgentConfig,
        search_tools: SearchTools
    ) -> Agent:
    tools = [search_tools.search, search_tools.get_file]
    
    search_agent = Agent(
        name=config.name,
        model=config.model,
        instructions=config.instructions,
        tools=tools
    )

    return search_agent



class NamedCallback:

    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)


async def run_agent(
        agent: Agent,
        user_prompt: str,
        message_history=None
    ) -> AgentRunResult:
    callback = NamedCallback(agent) 

    if message_history is None:
        message_history = []

    result = await agent.run(
        user_prompt,
        event_stream_handler=callback,
        message_history=message_history
    )

    return result





