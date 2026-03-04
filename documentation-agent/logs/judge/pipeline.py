import time
import json
from typing import List, Optional
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from logs.sql import SQLiteStorage, LogRecordRow, LogEventRow
from logs.models import LogRecord, LogEvent
from logs.judge.agent import create_judge_agent, JUDGE_PROMPT_TEMPLATE
from tools import SearchTools
from tests.cost_tracker import calculate_cost
from pydantic_ai.messages import ModelMessagesTypeAdapter
from models import RAGResponse

def get_agent_model_name(agent) -> str:
    """Extract the model name string from a pydantic_ai Agent."""
    try:
        model = agent.model
        # Use direct attribute access as requested
        return f"{model.system}:{model.model_name}"
    except Exception:
        return "openai:gpt-4o-mini"

def calculate_run_cost(agent, usage) -> float:
    """Calculate the cost of an agent run based on its usage."""
    model_name = get_agent_model_name(agent)
    return calculate_cost(model_name, usage.request_tokens, usage.response_tokens)

def extract_tool_calls(messages_json_str: str) -> List[dict]:
    """Extract tool calls and their arguments from the session messages."""
    try:
        messages = ModelMessagesTypeAdapter.validate_json(messages_json_str.encode('utf-8'))
        tool_calls = []
        for m in messages:
            # Check for ModelRequest which contains tool calls in pydantic_ai
            if hasattr(m, 'parts'):
                for p in m.parts:
                    if p.part_kind == 'tool-call':
                        # Skip the final_result tool as it's the output wrapper
                        if p.tool_name != 'final_result':
                            tool_calls.append({
                                "tool": p.tool_name,
                                "args": p.args
                            })
        return tool_calls
    except Exception as e:
        print(f"Error extracting tool calls: {e}")
        return []

class EvaluationPipeline:
    def __init__(self, storage: SQLiteStorage, search_tools: SearchTools):
        self.storage = storage
        self.search_tools = search_tools
        self.judge_agent = create_judge_agent(search_tools)

    def get_unevaluated_sessions(self, limit: int = 10) -> List[str]:
        """Find session IDs that have log records but no 'evaluation' event."""
        query = text("""
            SELECT DISTINCT lr.session_id 
            FROM log_records lr
            LEFT JOIN log_events le ON lr.session_id = le.session_id AND le.event_type = 'evaluation'
            WHERE le.id IS NULL
            LIMIT :limit
        """)
        
        with Session(self.storage.engine) as session:
            result = session.execute(query, {"limit": limit})
            return [row[0] for row in result]

    async def evaluate_session(self, session_id: str):
        """Evaluate all interactions in a given session."""
        print(f"Evaluating session: {session_id}")
        
        # Load logs for this session
        with Session(self.storage.engine) as session:
            stmt = select(LogRecordRow).where(LogRecordRow.session_id == session_id).order_by(LogRecordRow.timestamp)
            rows = session.execute(stmt).scalars().all()
            
            if not rows:
                print(f"No logs found for session {session_id}")
                return

            # For now, we evaluate the last interaction in the session
            # or we could aggregate them. Let's start with the last one.
            last_row = rows[-1]
            
            # Extract info for the judge
            # We need to parse messages to find the user prompt
            messages = json.loads(last_row.messages or "[]")
            user_message = "No user message found"
            for m in messages:
                if m.get('role') == 'user':
                    user_message = m.get('content', "No content")
                    break
            
            # Extract agent answer and use to_string if it's a RAGResponse
            agent_answer_raw = last_row.output
            agent_answer_str = agent_answer_raw
            
            # If output_type is RAGResponse OR it looks like JSON with 'answer' field
            if last_row.output_type == 'RAGResponse' or (agent_answer_raw and agent_answer_raw.strip().startswith('{')):
                try:
                    rag_data = json.loads(agent_answer_raw)
                    if isinstance(rag_data, dict) and 'answer' in rag_data:
                        rag_response = RAGResponse.model_validate(rag_data)
                        agent_answer_str = rag_response.to_string()
                except Exception as e:
                    # If it's not a valid RAGResponse, just keep the raw output
                    pass
            
            # Extract actual tool calls with arguments
            actual_tool_calls = extract_tool_calls(last_row.messages or "[]")
            tool_calls_str = json.dumps(actual_tool_calls, indent=2) if actual_tool_calls else "No tool calls found"
            
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                user_message=user_message,
                agent_answer=agent_answer_str,
                tool_calls=tool_calls_str
            )

            print(f"Judge prompt:\n{prompt}")
            
            try:
                result = await self.judge_agent.run(prompt)
                
                # Capture usage and cost
                usage = result.usage()
                cost = calculate_run_cost(self.judge_agent, usage)
                
                evaluation_data = result.output.model_dump()
                evaluation_data["cost"] = cost
                
                # Store the evaluation as an event
                event = LogEvent(
                    session_id=session_id,
                    timestamp=time.time(),
                    event_type="evaluation",
                    event_data=evaluation_data
                )

                self.storage.save_event(event)
                print(f"Successfully evaluated session {session_id}. Score: {evaluation_data['overall_score']}, Cost: ${cost:.6f}")
            except Exception as e:
                print(f"Error evaluating session {session_id}: {e}")

    async def run_once(self):
        sessions = self.get_unevaluated_sessions()
        if not sessions:
            print("No unevaluated sessions found.")
            return

        for session_id in sessions:
            await self.evaluate_session(session_id)
