import sys
import os

def is_running_under_pytest():
    return "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None