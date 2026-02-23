
import dotenv
dotenv.load_dotenv()


from tests.cost_tracker import display_total_usage, reset_cost_file

def pytest_sessionstart(session):
    reset_cost_file()

def pytest_sessionfinish(session, exitstatus):
    display_total_usage()
