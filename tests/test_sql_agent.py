"""Tests for the SQL Agent.

This module contains tests for the SQL Agent functionality.
"""

import os
import tempfile
from typing import Generator
import pytest
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, text
import pandas as pd

from ember.examples.sql_agent.sql_agent import SQLAgent
from ember.examples.sql_agent.load_f1_data import load_f1_data

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file for testing.
    
    Returns:
        A path to a temporary database file.
    """
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
        temp_db_path = temp_db.name
        
    try:
        yield temp_db_path
    finally:
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)


@pytest.fixture
def test_db(temp_db_path: str) -> Generator[str, None, None]:
    """Create a test database with sample data.
    
    Args:
        temp_db_path: Path to the temporary database file.
        
    Returns:
        The path to the test database.
    """
    # Create a simple test database with one table
    engine = create_engine(f"sqlite:///{temp_db_path}")
    
    # Create a simple test table
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER
        )
        """))
        
        # Insert some test data
        conn.execute(text("""
        INSERT INTO test_table (id, name, value) VALUES 
        (1, 'Item 1', 100),
        (2, 'Item 2', 200),
        (3, 'Item 3', 300)
        """))
        
        conn.commit()
    
    yield temp_db_path


@pytest.fixture
def f1_test_db(temp_db_path: str) -> Generator[str, None, None]:
    """Create a test database with Formula 1 sample data.
    
    Args:
        temp_db_path: Path to the temporary database file.
        
    Returns:
        The path to the test database with F1 data.
    """
    # Load the F1 data into the temporary database
    load_f1_data(temp_db_path)
    yield temp_db_path


@pytest.fixture
def mock_model(monkeypatch: "MonkeyPatch") -> None:
    """Mock the Ember model to avoid actual API calls.
    
    Args:
        monkeypatch: pytest monkeypatch fixture.
    """
    def mock_model_factory(*args, **kwargs):
        def mock_model_call(prompt: str) -> str:
            if "write a SQL query" in prompt:
                return "SELECT * FROM test_table WHERE value > 100"
            elif "rewrite the query" in prompt:
                return "SELECT * FROM test_table WHERE value > 150"
            elif "Answer the following question" in prompt:
                return "There are 2 items with values greater than 100: Item 2 (200) and Item 3 (300)."
            return "Mocked response"
        return mock_model_call
    
    monkeypatch.setattr("ember.api.models.model", mock_model_factory)


def test_sql_agent_initialization() -> None:
    """Test that the SQL Agent initializes correctly."""
    # This test just verifies the class can be instantiated without errors
    agent = SQLAgent(database_url="sqlite:///:memory:")
    assert agent is not None
    assert agent.database_url == "sqlite:///:memory:"


def test_sql_agent_query_execution(test_db: str, mock_model: None) -> None:
    """Test that the SQL Agent can execute queries.
    
    Args:
        test_db: Path to the test database.
        mock_model: Mocked model fixture.
    """
    # Initialize the agent with the test database
    agent = SQLAgent(database_url=f"sqlite:///{test_db}")
    
    # Execute a query
    result = agent.query("What items have a value greater than 100?")
    
    # Check that the response has the expected structure
    assert "question" in result
    assert "sql_query" in result
    assert "query_result" in result
    assert "answer" in result
    
    # Check the query result
    assert result["query_result"]["success"] is True
    assert result["query_result"]["record_count"] == 2
    
    # Check that the records are correct
    records = result["query_result"]["results"]
    assert len(records) == 2
    assert records[0]["name"] == "Item 2"
    assert records[0]["value"] == 200
    assert records[1]["name"] == "Item 3"
    assert records[1]["value"] == 300


def test_schema_exploration(test_db: str, mock_model: None) -> None:
    """Test that the SQL Agent can explore the database schema.
    
    Args:
        test_db: Path to the test database.
        mock_model: Mocked model fixture.
    """
    # Initialize the agent with the test database
    agent = SQLAgent(database_url=f"sqlite:///{test_db}")
    
    # Access the schema
    schema = agent._get_database_schema()
    
    # Check the schema structure
    assert "tables" in schema
    assert "test_table" in schema["tables"]
    
    # Check the table info
    table_info = schema["tables"]["test_table"]
    assert "columns" in table_info
    assert "sample_data" in table_info
    assert "row_count" in table_info
    
    # Check the columns
    columns = table_info["columns"]
    assert len(columns) == 3
    column_names = [col["name"] for col in columns]
    assert "id" in column_names
    assert "name" in column_names
    assert "value" in column_names
    
    # Check the row count
    assert table_info["row_count"] == 3
    
    # Check the sample data
    sample_data = table_info["sample_data"]
    assert len(sample_data) == 3 