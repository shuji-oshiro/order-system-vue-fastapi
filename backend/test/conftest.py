# backend/test/conftest.py
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_memory_db():
    os.environ["PYTEST_RUNNING"] = "1"  # テスト実行中のフラグを設定        
    yield


