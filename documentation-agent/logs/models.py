from time import time
from typing import Generic, Optional, TypeVar
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunUsage, AgentRunResult
from pydantic_ai.messages import ModelMessage

T = TypeVar("T", str, BaseModel)


@dataclass
class AgentInfo:
    name: str
    model: str
    instructions: str
    tools: list[str]


@dataclass
class LogRecord(Generic[T]):
    session_id: str
    timestamp: float
    agent_info: AgentInfo
    messages: list[ModelMessage]
    usage: RunUsage
    time_to_first_token: float
    execution_time: float
    output: T


@dataclass
class LogEvent:
    session_id: str
    timestamp: float
    event_type: str
    event_data: dict



def create_log_record(
    session_id: str,
    agent: Agent,
    run_result: AgentRunResult[T],
    execution_time: float,
    time_to_first_token: float
) -> LogRecord[T]:
    tools = []

    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())

    provider = agent.model.system
    model_name = agent.model.model_name
    model = f'{provider}:{model_name}'

    agent_info = AgentInfo(
        name=agent.name,
        instructions='\n'.join(agent._instructions),
        model=model,
        tools=tools
    )

    return LogRecord(
        session_id=session_id,
        timestamp=time(),
        agent_info=agent_info,
        messages=run_result.new_messages(),
        usage=run_result.usage(),
        output=run_result.output,
        execution_time=execution_time,
        time_to_first_token=time_to_first_token
    )