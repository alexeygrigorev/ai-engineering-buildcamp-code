import json
import time
from typing import Optional, List, Union, Tuple
from sqlalchemy import select
from sqlalchemy.orm import Session

from logs.models import LogRecord, LogEvent, AgentInfo, RunUsage
from logs.sql import LogRecordRow, LogEventRow, ModelMessagesTypeAdapter


def parse_period(
    period: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses a period into (start_time, end_time) timestamps.
    - period can be a string: 'today', 'last_hour', 'last_30_minutes'
    - period can be a tuple: (start_time, end_time)
    """
    now = time.time()
    start_time, end_time = None, None

    if isinstance(period, str):
        if period == "today":
            start_time = now - 24 * 3600
        elif period == "last_hour":
            start_time = now - 3600
        elif period == "last_30_minutes":
            start_time = now - 1800
    elif isinstance(period, tuple):
        start_time, end_time = period

    return start_time, end_time


class LogLoader:
    """Base class for log and event loaders."""

    def load_logs(
        self,
        period: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[LogRecord]:
        raise NotImplementedError

    def load_events(
        self,
        event_type: Optional[str] = None,
        period: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[LogEvent]:
        raise NotImplementedError


class SQLLogLoader(LogLoader):
    """SQL-based implementation of LogLoader."""

    def __init__(self, engine):
        self.engine = engine

    def load_logs(
        self,
        period: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[LogRecord]:
        start_time, end_time = parse_period(period)

        stmt = select(LogRecordRow)
        if start_time is not None:
            stmt = stmt.where(LogRecordRow.timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(LogRecordRow.timestamp <= end_time)

        with Session(self.engine) as session:
            rows = session.execute(stmt).scalars().all()
            logs = []
            for row in rows:
                ai = AgentInfo(
                    name=row.agent_info__name,
                    model=row.agent_info__model,
                    instructions=row.agent_info__instructions,
                    tools=json.loads(row.agent_info__tools or "[]"),
                )
                u = RunUsage(
                    requests=row.usage__requests or 0,
                    input_tokens=row.usage__request_tokens or 0,
                    output_tokens=row.usage__response_tokens or 0,
                )

                logs.append(
                    LogRecord(
                        session_id=row.session_id,
                        timestamp=row.timestamp,
                        agent_info=ai,
                        messages=ModelMessagesTypeAdapter.validate_json(
                            row.messages.encode("utf-8")
                        ),
                        usage=u,
                        time_to_first_token=row.time_to_first_token,
                        execution_time=row.execution_time,
                        output=row.output,
                    )
                )
            return logs

    def load_events(
        self,
        event_type: Optional[str] = None,
        period: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[LogEvent]:
        start_time, end_time = parse_period(period)

        stmt = select(LogEventRow)
        if event_type is not None:
            stmt = stmt.where(LogEventRow.event_type == event_type)
        if start_time is not None:
            stmt = stmt.where(LogEventRow.timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(LogEventRow.timestamp <= end_time)

        with Session(self.engine) as session:
            rows = session.execute(stmt).scalars().all()
            return [
                LogEvent(
                    session_id=row.session_id,
                    timestamp=row.timestamp,
                    event_type=row.event_type,
                    event_data=json.loads(row.event_data or "{}"),
                )
                for row in rows
            ]
