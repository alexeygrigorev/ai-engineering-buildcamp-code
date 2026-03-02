from uuid import uuid4
from time import time
from pydantic_ai import Agent, AgentRunResult
from logs.models import LogRecord, LogEvent, create_log_record, T


class Storage:
    def save_log(self, log_record: LogRecord):
        pass

    def save_event(self, event: LogEvent):
        pass


class NoOpStorage(Storage):
    def save_log(self, log_record: LogRecord):
        print(f'not saving {log_record}...')

    def save_event(self, event: LogEvent):
        print(f'not saving event {event}...')



class MonitoringService:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.session_id = str(uuid4())

    def restart_session(self):
        self.session_id = str(uuid4())

    def log_agent_run(self,
        agent: Agent,
        run_result: AgentRunResult[T],
        execution_time: float,
        time_to_first_token: float
    ):
        log_record = create_log_record(
            self.session_id,
            agent,
            run_result,
            execution_time,
            time_to_first_token
        )
        self.storage.save_log(log_record)

    def log_event(self, event_type: str, event_data: dict):
        event = LogEvent(
            session_id=self.session_id,
            timestamp=time(),
            event_type=event_type,
            event_data=event_data
        )
        self.storage.save_event(event)