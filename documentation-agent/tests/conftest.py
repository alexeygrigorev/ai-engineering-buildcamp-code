
from tests.cost_tracker import display_total_usage

def pytest_sessionfinish(session, exitstatus):
    display_total_usage()
