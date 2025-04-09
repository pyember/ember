# SQL Agent Example

## Overview
This example demonstrates a dynamic SQL Agent capable of handling any database schema without hardcoded patterns. The agent can dynamically explore and map database structures, generate appropriate SQL queries, and provide definitive answers for any question.

## Key Features

### 1. Dynamic Database Schema Exploration
- Functionality to dynamically explore the entire database schema
- Retrieves all tables, columns, and sample data
- Attempts to infer relationships between tables
- Adapts to any database structure without hardcoding

### 2. Improved Query Generation
- Generates SQL queries based on the complete database schema
- Explicitly instructs the LLM to use actual table names, not placeholders
- Validates queries against the actual database schema
- Automatically corrects queries with invalid column or table names

### 3. Enhanced Answer Generation
- Generates clear, definitive answers based on query results and database schema
- Includes specific details from query results in the answers
- Adapts to any type of question without relying on predefined patterns
- Distills SQL query results and adds them to the LLM context

## Usage

### Command-line Interface
```python
from ember.examples.sql_agent.sql_agent import SQLAgent

# Initialize the agent with a database connection
agent = SQLAgent(database_url="sqlite:///your_database.db")

# Ask a question about your data
result = agent.query("What was the average value in the last month?")
print(result["answer"])
```

### Streamlit Web Interface
The SQL Agent also comes with a Streamlit web interface for interactive querying:

```bash
# First, set up your environment with the required API key(s)
export OPENAI_API_KEY='your-api-key'  # or ANTHROPIC_API_KEY, DEEPMIND_API_KEY

# Then run the Streamlit app
streamlit run src/ember/examples/sql_agent/app.py
```

This launches a web interface where you can:
- Ask questions about your data in natural language
- See the generated SQL queries and their results
- Browse sample queries
- Select different LLM models
- Export your chat history

## Example Data
This example includes a Formula 1 dataset with tables for:
- Drivers championship
- Constructors championship
- Race results
- Race wins
- Fastest laps

## Technical Details
- Uses SQLAlchemy's `text()` function for safer SQL query execution
- Implements dynamic schema mapping to understand any database structure
- Adds query validation and correction mechanisms
- Enhances the answer generation process to provide more definitive responses

## PR Enhancement Notes
This SQL Agent example was created as part of a PR to enhance the Ember framework with a dynamic SQL Agent that can work with any database schema without hardcoded patterns. The implementation follows best practices for Python development with proper typing, documentation, and test coverage. 