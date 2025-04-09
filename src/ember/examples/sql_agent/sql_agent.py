"""SQL Agent Module.

This module provides a dynamic SQL Agent capable of understanding any database schema
and answering questions about the data.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from ember.api import models
from ember.examples.sql_agent.utils import add_message

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLAgent:
    """A dynamic SQL Agent that can work with any database schema.
    
    This agent can dynamically explore database schemas, generate appropriate SQL queries,
    and provide definitive answers for any question about the data.
    
    Attributes:
        database_url: The SQLAlchemy connection URL for the database.
        model_name: The LLM model to use for query generation and answer synthesis.
        temperature: The temperature setting for the LLM.
    """

    def __init__(self, 
                 database_url: str = "sqlite:///f1_data.db", 
                 model_name: str = "openai:gpt-4o", 
                 temperature: float = 0.0) -> None:
        """Initialize the SQL Agent.
        
        Args:
            database_url: SQLAlchemy connection URL string.
            model_name: Model identifier to use for SQL generation and answer synthesis.
                Options include:
                - OpenAI: "openai:gpt-4o", "openai:gpt-4", "openai:gpt-3.5-turbo"
                - Anthropic: "anthropic:claude-3-opus", "anthropic:claude-3-sonnet" 
                - Deepmind: "deepmind:gemini-1.5-pro"
            temperature: Temperature setting for the model (0.0 for deterministic outputs).
        """
        self.database_url = database_url
        self.model_name = model_name
        self.temperature = temperature
        
        # Create database engine
        self.engine = create_engine(database_url)
        
        # Create the model
        self.model = models.model(model_name, temperature=temperature)
        
        # Cache for database schema
        self._schema_cache = None
        
    def _get_database_schema(self) -> Dict[str, Any]:
        """Dynamically retrieve and cache the database schema.
        
        Returns:
            A dictionary containing the database schema information.
        """
        if self._schema_cache:
            return self._schema_cache
            
        logger.info("Retrieving database schema")
        inspector = inspect(self.engine)
        
        schema = {
            "tables": {}
        }
        
        # Get all tables
        for table_name in inspector.get_table_names():
            table_info = {
                "columns": [],
                "sample_data": None,
                "row_count": 0
            }
            
            # Get column information
            for column in inspector.get_columns(table_name):
                table_info["columns"].append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column.get("nullable", True)
                })
            
            # Get sample data
            try:
                query = f"SELECT * FROM {table_name} LIMIT 5"
                sample_df = pd.read_sql(query, self.engine)
                table_info["sample_data"] = sample_df.to_dict(orient="records")
                
                # Get row count
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_df = pd.read_sql(count_query, self.engine)
                table_info["row_count"] = int(count_df["count"].iloc[0])
            except Exception as e:
                logger.error(f"Error getting sample data for {table_name}: {str(e)}")
            
            schema["tables"][table_name] = table_info
        
        # Cache the schema
        self._schema_cache = schema
        return schema
        
    def _generate_sql_query(self, question: str) -> str:
        """Generate a SQL query to answer the given question.
        
        Args:
            question: The question to generate SQL for.
            
        Returns:
            A SQL query string.
        """
        schema = self._get_database_schema()
        
        # Format schema info for prompt
        schema_info = "Database Schema:\n"
        for table_name, table_info in schema["tables"].items():
            schema_info += f"Table: {table_name} ({table_info['row_count']} rows)\n"
            schema_info += "Columns:\n"
            
            for column in table_info["columns"]:
                nullable = "NULL" if column["nullable"] else "NOT NULL"
                schema_info += f"  - {column['name']} ({column['type']}) {nullable}\n"
            
            # Include sample data
            if table_info["sample_data"]:
                schema_info += "Sample Data:\n"
                sample_df = pd.DataFrame(table_info["sample_data"])
                schema_info += f"{sample_df.head().to_string()}\n\n"
        
        # Prompt for SQL generation
        prompt = f"""You are an expert SQL developer. Given the following database schema, write a SQL query to answer this question: "{question}"

{schema_info}

Rules:
1. Use only the tables and columns that exist in the schema
2. Write a valid SQL query that will run on SQLite
3. Use table and column names exactly as they appear in the schema
4. Do not use any placeholders like <table_name> or <column_name>
5. Always return a query that will provide a meaningful answer
6. Format complex queries for readability

SQL query:"""

        # Generate the SQL query
        start_time = time.time()
        generated_sql = self.model(prompt).strip()
        end_time = time.time()
        logger.info(f"Query generation took {end_time - start_time:.2f} seconds")
        
        return generated_sql
        
    def _validate_and_correct_query(self, query: str) -> str:
        """Validate and correct SQL queries before execution.
        
        Args:
            query: The SQL query to validate.
            
        Returns:
            A corrected SQL query string.
        """
        schema = self._get_database_schema()
        
        # Check for table names
        tables = schema["tables"].keys()
        for table in tables:
            # Very basic check - could be enhanced
            if table in query:
                return query
        
        # If no valid tables found, try to correct
        prompt = f"""The following SQL query doesn't seem to reference any existing tables in our database:

{query}

Our database has the following tables: {', '.join(tables)}

Please rewrite the query to use the correct table names:"""

        corrected_query = self.model(prompt).strip()
        return corrected_query
        
    def _execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a SQL query and return the results.
        
        Args:
            query: The SQL query to execute.
            
        Returns:
            A dictionary with the query execution results and metadata.
        """
        try:
            # Validate and correct query
            validated_query = self._validate_and_correct_query(query)
            
            # Execute query
            start_time = time.time()
            result_df = pd.read_sql(text(validated_query), self.engine)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Convert results to dictionary
            records = result_df.to_dict(orient="records")
            
            return {
                "success": True,
                "query": validated_query,
                "results": records,
                "record_count": len(records),
                "execution_time": f"{execution_time:.2f} seconds",
                "columns": list(result_df.columns),
                "error": None
            }
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                "success": False,
                "query": query,
                "results": [],
                "record_count": 0,
                "execution_time": "0.00 seconds",
                "columns": [],
                "error": str(e)
            }
    
    def _generate_answer(self, question: str, query_result: Dict[str, Any]) -> str:
        """Generate a natural language answer based on the query results.
        
        Args:
            question: The original question.
            query_result: The results from the query execution.
            
        Returns:
            A natural language answer to the question.
        """
        # Format the results for the prompt
        formatted_results = "No results found."
        if query_result["success"] and query_result["results"]:
            result_df = pd.DataFrame(query_result["results"])
            formatted_results = result_df.to_string()
        
        # Prompt for answer generation
        prompt = f"""You are an analytics assistant. Answer the following question based on the database query results.

Question: {question}

SQL Query: {query_result["query"]}

Query Results:
{formatted_results}

Execution Information:
- Record Count: {query_result["record_count"]}
- Execution Time: {query_result["execution_time"]}
- Success: {query_result["success"]}
{f'- Error: {query_result["error"]}' if not query_result["success"] else ''}

Instructions:
1. Provide a clear, definitive answer to the question
2. Include specific details and numbers from the query results
3. If the query failed, explain why and suggest a better approach
4. Be concise but thorough
5. Format the answer for readability with markdown where appropriate

Answer:"""

        # Generate the answer
        start_time = time.time()
        answer = self.model(prompt).strip()
        end_time = time.time()
        logger.info(f"Answer generation took {end_time - start_time:.2f} seconds")
        
        return answer
        
    def query(self, question: str) -> Dict[str, Any]:
        """Process a natural language question about the database.
        
        Args:
            question: The question to answer.
            
        Returns:
            A dictionary containing the question, SQL query, results, and answer.
        """
        logger.info(f"Processing question: {question}")
        
        # Generate SQL query
        sql_query = self._generate_sql_query(question)
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Execute the query
        query_result = self._execute_query(sql_query)
        
        # Generate answer
        answer = self._generate_answer(question, query_result)
        
        # Return complete response
        return {
            "question": question,
            "sql_query": sql_query,
            "query_result": query_result,
            "answer": answer
        }
        
    def chat(self, question: str, messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Interactive chat interface with message history.
        
        Args:
            question: The question to answer.
            messages: Optional list of previous messages.
            
        Returns:
            A dictionary with the updated message history and latest response.
        """
        if messages is None:
            messages = []
            
        # Add user question to messages
        messages = add_message(messages, "user", question)
        
        # Process the question
        response = self.query(question)
        
        # Create assistant message content
        content = response["answer"]
        
        # Create tool calls list for transparency
        tool_calls = [
            {
                "tool_name": "generate_sql_query",
                "tool_args": {"question": question},
                "content": response["sql_query"]
            },
            {
                "tool_name": "execute_sql_query",
                "tool_args": {"query": response["sql_query"]},
                "content": json.dumps(response["query_result"], indent=2)
            }
        ]
        
        # Add assistant response to messages
        messages = add_message(messages, "assistant", content, tool_calls)
        
        return {
            "messages": messages,
            "response": response
        } 