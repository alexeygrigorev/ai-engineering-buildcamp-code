import json
import os
from pathlib import Path
from typing import Optional, Any, List, Union, Tuple
from sqlalchemy import create_engine, Float, Integer, String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessagesTypeAdapter

from logs.models import LogRecord, LogEvent
from logs.service import Storage


class Base(DeclarativeBase):
    pass


class LogRecordRow(Base):
    """ORM model for the log_records table."""

    __tablename__ = "log_records"

    id: Mapped[Optional[int]] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    agent_info__name: Mapped[Optional[str]] = mapped_column(String)
    agent_info__model: Mapped[Optional[str]] = mapped_column(String)
    agent_info__instructions: Mapped[Optional[str]] = mapped_column(String)
    agent_info__tools: Mapped[Optional[str]] = mapped_column(String)  # JSON list
    messages: Mapped[Optional[str]] = mapped_column(String)  # JSON list
    usage__requests: Mapped[Optional[int]] = mapped_column(Integer)
    usage__request_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    usage__response_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    usage__total_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    time_to_first_token: Mapped[Optional[float]] = mapped_column(Float)
    execution_time: Mapped[Optional[float]] = mapped_column(Float)
    output: Mapped[Optional[str]] = mapped_column(String)
    output_type: Mapped[Optional[str]] = mapped_column(String)


class LogEventRow(Base):
    """ORM model for the log_events table."""

    __tablename__ = "log_events"

    id: Mapped[Optional[int]] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    event_data: Mapped[Optional[str]] = mapped_column(String)  # JSON dict


class SQLStorage(Storage):
    def __init__(self, engine):
        self.engine = engine
        Base.metadata.create_all(self.engine)

        from logs.loaders import SQLLogLoader

        self.loader = SQLLogLoader(self.engine)

    def save_log(self, log_record: LogRecord):
        ai = log_record.agent_info
        u = log_record.usage

        messages_json = ModelMessagesTypeAdapter.dump_json(log_record.messages).decode(
            "utf-8"
        )

        output = log_record.output
        output_str = (
            output.model_dump_json() if isinstance(output, BaseModel) else str(output)
        )

        row = LogRecordRow(
            session_id=log_record.session_id,
            timestamp=log_record.timestamp,
            agent_info__name=ai.name,
            agent_info__model=ai.model,
            agent_info__instructions=ai.instructions,
            agent_info__tools=json.dumps(ai.tools),
            messages=messages_json,
            usage__requests=u.requests,
            usage__request_tokens=u.input_tokens,
            usage__response_tokens=u.output_tokens,
            usage__total_tokens=u.total_tokens,
            time_to_first_token=log_record.time_to_first_token,
            execution_time=log_record.execution_time,
            output=output_str,
            output_type=type(output).__name__,
        )

        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def save_event(self, event: LogEvent):
        row = LogEventRow(
            session_id=event.session_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            event_data=json.dumps(event.event_data),
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def load_logs(
        self,
        period: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[LogRecord]:
        return self.loader.load_logs(period=period)

    def load_events(
        self,
        event_type: Optional[str] = None,
        period: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[LogEvent]:
        return self.loader.load_events(event_type=event_type, period=period)


class SQLiteStorage(SQLStorage):
    def __init__(self, db_path: str = "db/logs.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}")
        super().__init__(engine)


class PostgresStorage(SQLStorage):
    def __init__(self, url: str):
        engine = create_engine(url)
        super().__init__(engine)


def get_storage() -> Storage:
    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("postgresql"):
        return PostgresStorage(database_url)
    
    db_path = os.getenv("SQLITE_DB_PATH", "db/logs.db")
    return SQLiteStorage(db_path)
