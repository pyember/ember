import json
import os
from pathlib import Path
from textwrap import dedent
import logging
from typing import Dict, ClassVar, Optional

from ember.api import models
from ember.api.operators import Operator, Specification, EmberModel, Field
from sqlalchemy import create_engine, text
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection string
DB_URL = "sqlite:///f1_data.db"

# Paths
CWD = Path(__file__).parent
KNOWLEDGE_DIR = CWD.joinpath("knowledge")
OUTPUT_DIR = CWD.joinpath("output")

# Create the output directory if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define input and output models
class SQLQueryInput(EmberModel):
    query: str = Field(..., description="The natural language query to process")

class SQLQueryOutput(EmberModel):
    response: str = Field(..., description="The generated response")
    sql_query: Optional[str] = Field(None, description="The SQL query that was executed")
    execution_time: Optional[float] = Field(None, description="Time taken to execute the query in seconds")

# Define specification
class SQLAgentSpec(Specification):
    input_model: type[EmberModel] = SQLQueryInput
    structured_output: type[EmberModel] = SQLQueryOutput

class SQLAgent(Operator[SQLQueryInput, SQLQueryOutput]):
    """SQL Agent that converts natural language to SQL and executes queries."""

    specification: ClassVar[Specification] = SQLAgentSpec()

    def __init__(self, model_name: str = "openai:gpt-4o"):
        """Initialize the SQL Agent.

        Args:
            model_name: Model identifier in format 'provider:model_name'
                Options include:
                - OpenAI: "openai:gpt-4o", "openai:gpt-4o-mini", "openai:gpt-4", "openai:gpt-4-turbo", "openai:gpt-3.5-turbo"
                - Anthropic: "anthropic:claude-3-opus", "anthropic:claude-3-5-sonnet", "anthropic:claude-3-5-haiku", "anthropic:claude-3-7-sonnet"
                - Deepmind: "deepmind:gemini-1.5-pro", "deepmind:gemini-1.5-flash", "deepmind:gemini-2.0-pro"
        """
        super().__init__()
        self.model_id = model_name
        self.llm = models.model(model_name, temperature=0.2)
        self.db_engine = create_engine(DB_URL)
        self.knowledge_base = self._load_knowledge_base()
        self.semantic_model = self._create_semantic_model()
        self.session_name = "New Session"
        self.messages = []
        self.tool_calls = []

    def _load_knowledge_base(self) -> Dict:
        """Load the knowledge base from the knowledge directory."""
        knowledge_base = {
            "tables": {},
            "sample_queries": []
        }

        # Load table metadata
        for file_path in KNOWLEDGE_DIR.glob("*.json"):
            with open(file_path, "r") as f:
                table_data = json.load(f)
                table_name = table_data.get("table_name")
                if table_name:
                    knowledge_base["tables"][table_name] = table_data

        # Load sample queries
        sample_queries_path = KNOWLEDGE_DIR / "sample_queries.sql"
        if sample_queries_path.exists():
            with open(sample_queries_path, "r") as f:
                content = f.read()

                # Parse the sample queries
                query_blocks = []
                current_block = {"description": "", "query": ""}
                in_description = False
                in_query = False

                for line in content.split("\n"):
                    if line.strip() == "-- <query description>":
                        in_description = True
                        in_query = False
                        if current_block["query"]:
                            query_blocks.append(current_block)
                            current_block = {"description": "", "query": ""}
                    elif line.strip() == "-- </query description>":
                        in_description = False
                    elif line.strip() == "-- <query>":
                        in_query = True
                        in_description = False
                    elif line.strip() == "-- </query>":
                        in_query = False
                    elif in_description:
                        current_block["description"] += line.replace("-- ", "") + "\n"
                    elif in_query:
                        current_block["query"] += line + "\n"

                if current_block["query"]:
                    query_blocks.append(current_block)

                knowledge_base["sample_queries"] = query_blocks

        return knowledge_base

    def _create_semantic_model(self) -> Dict:
        """Create a semantic model from the knowledge base."""
        semantic_model = {
            "tables": []
        }

        for table_name, table_data in self.knowledge_base["tables"].items():
            semantic_model["tables"].append({
                "table_name": table_name,
                "table_description": table_data.get("table_description", ""),
                "Use Case": f"Use this table to get data on {table_name.replace('_', ' ')}."
            })

        return semantic_model

    def search_knowledge_base(self, table_name: str) -> str:
        """Search the knowledge base for information about a table."""
        # Log the tool call
        self.tool_calls.append({
            "tool_name": "search_knowledge_base",
            "tool_args": {"table_name": table_name},
            "content": None
        })

        if table_name in self.knowledge_base["tables"]:
            table_data = self.knowledge_base["tables"][table_name]
            result = {
                "table_name": table_data.get("table_name", ""),
                "table_description": table_data.get("table_description", ""),
                "table_columns": table_data.get("table_columns", []),
                "table_rules": table_data.get("table_rules", [])
            }
            result_str = json.dumps(result, indent=2)
            self.tool_calls[-1]["content"] = result_str
            return result_str
        else:
            result = f"Table '{table_name}' not found in the knowledge base."
            self.tool_calls[-1]["content"] = result
            return result

    def describe_table(self, table_name: str) -> str:
        """Get the schema of a table from the database."""
        # Log the tool call
        self.tool_calls.append({
            "tool_name": "describe_table",
            "tool_args": {"table_name": table_name},
            "content": None
        })

        try:
            # For SQLite, we need to use a different approach to get table schema
            # since it doesn't support information_schema
            query = f"PRAGMA table_info({table_name});"

            with self.db_engine.connect() as conn:
                result = conn.execute(text(query))
                columns = result.fetchall()

            if not columns:
                result = f"Table '{table_name}' not found in the database."
                self.tool_calls[-1]["content"] = result
                return result

            schema_info = f"Schema for table '{table_name}':\n\n"
            schema_info += "| Column Name | Data Type | Nullable | Primary Key |\n"
            schema_info += "|-------------|-----------|----------|------------|\n"

            for column in columns:
                # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
                col_name = column[1]
                col_type = column[2]
                col_nullable = "NO" if column[3] == 1 else "YES"
                col_pk = "YES" if column[5] == 1 else "NO"

                schema_info += f"| {col_name} | {col_type} | {col_nullable} | {col_pk} |\n"

            # Also get a sample of data to help understand the table
            sample_query = f"SELECT * FROM {table_name} LIMIT 3;"
            try:
                with self.db_engine.connect() as conn:
                    sample_result = conn.execute(text(sample_query))
                    sample_rows = sample_result.fetchall()
                    sample_columns = sample_result.keys()

                    if sample_rows:
                        schema_info += f"\n\nSample data from '{table_name}':\n\n"
                        # Create header row
                        schema_info += "| " + " | ".join(sample_columns) + " |\n"
                        # Create separator row
                        schema_info += "| " + " | ".join(["---" for _ in sample_columns]) + " |\n"
                        # Add data rows
                        for row in sample_rows:
                            schema_info += "| " + " | ".join([str(cell) for cell in row]) + " |\n"
            except Exception as e:
                schema_info += f"\n\nCould not retrieve sample data: {str(e)}"

            self.tool_calls[-1]["content"] = schema_info
            return schema_info

        except Exception as e:
            error_message = f"Error describing table: {str(e)}"
            logger.error(error_message)
            self.tool_calls[-1]["content"] = error_message
            return error_message

    def run_sql_query(self, query: str) -> str:
        """Run a SQL query and return the results as a formatted string."""
        try:
            # Log the query
            logger.info(f"Running SQL query: {query}")
            self.tool_calls.append({
                "tool_name": "run_sql_query",
                "tool_args": {"query": query},
                "content": None
            })

            # Execute the query
            with self.db_engine.connect() as conn:
                df = pd.read_sql(query, conn)

            # Convert to markdown table
            markdown_table = df.to_markdown(index=False)

            # Store the result in the tool call
            self.tool_calls[-1]["content"] = json.dumps(df.to_dict(orient="records"))

            return markdown_table

        except Exception as e:
            error_message = f"Error executing SQL query: {str(e)}"
            logger.error(error_message)
            self.tool_calls[-1]["content"] = json.dumps({"error": error_message})
            return error_message

    def get_tool_call_history(self, num_calls: int = 3) -> str:
        """Get the history of tool calls."""
        if not self.tool_calls:
            return "No tool calls in history."

        recent_calls = self.tool_calls[-num_calls:]
        result = "Recent tool calls:\n\n"

        for i, call in enumerate(recent_calls):
            result += f"Tool Call {i+1}:\n"
            result += f"Tool: {call['tool_name']}\n"

            if call['tool_name'] == 'run_sql_query':
                result += f"Query:\n```sql\n{call['tool_args']['query']}\n```\n"
            else:
                result += f"Arguments: {json.dumps(call['tool_args'], indent=2)}\n"

            result += "\n"

        return result

    def execute_sql_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return the results as a DataFrame.

        Args:
            query: The SQL query to execute

        Returns:
            A pandas DataFrame with the query results
        """
        logger.info(f"Executing SQL query: {query}")
        # Add this to the tool calls for tracking
        self.tool_calls.append({
            "tool_name": "execute_sql_query",
            "tool_args": {"query": query},
            "content": None
        })

        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                columns = result.keys()
                df = pd.DataFrame(rows, columns=columns)

                # Store the result in the tool call
                self.tool_calls[-1]["content"] = json.dumps(df.to_dict(orient="records"))

                return df
        except Exception as e:
            error_message = f"Error executing SQL query: {str(e)}"
            logger.error(error_message)
            self.tool_calls[-1]["content"] = json.dumps({"error": error_message})
            raise

    def forward(self, *, inputs: SQLQueryInput) -> SQLQueryOutput:
        """Process a natural language query and return a response.

        Args:
            inputs: The input query

        Returns:
            The generated response with SQL query information
        """
        query = inputs.query

        # Add the user message to the conversation history
        self.messages.append({"role": "user", "content": query})

        # Prepare the system prompt
        system_prompt = self._create_system_prompt()

        # Prepare the conversation history as a formatted string
        conversation_history = ""
        for message in self.messages[:-1]:  # Exclude the current query
            role = "User" if message["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {message['content']}\n\n"

        # Format the prompt
        prompt = f"""
        {system_prompt}

        CONVERSATION HISTORY:
        {conversation_history}

        USER QUERY: {query}

        IMPORTANT INSTRUCTIONS:
        1. ALWAYS use the describe_table() function to check the actual schema of any table before writing a query.
        2. You MUST include a SQL query in your response to answer the user's question.
        3. Format the SQL query as a code block with ```sql at the beginning and ``` at the end.
        4. Make sure your SQL query is correct and will run successfully.
        5. DO NOT include placeholders or comments in your SQL query that would prevent it from executing.
        6. DO NOT include multiple SQL queries - just provide ONE complete query that answers the question.
        7. ALWAYS provide a direct, clear answer at the beginning of your response.
        8. For questions about "most", "top", "longest", etc., clearly state the answer with specific values.
        9. ALWAYS execute the SQL query and include the results in your response.
        10. Be dynamic and flexible - handle ANY type of question about the data without relying on predefined patterns.
        11. NEVER stop after just retrieving schema information - ALWAYS proceed to generate and execute a SQL query.
        12. Your response MUST include a SQL query in a code block - this is critical for the system to work.
        13. CAREFULLY check the schema information to use the CORRECT column names in your query.
        14. Pay close attention to the sample data to understand the structure and content of each table.
        15. Generate SQL queries dynamically based on the question - don't rely on templates or patterns.
        16. Be creative in your SQL query construction to answer complex questions accurately.
        17. ALWAYS provide a clear, definitive answer based on the query results.

        For example, if the user asks "Who are the top 5 drivers with the most race wins?", your response should include:

        ```sql
        SELECT name, COUNT(*) as wins
        FROM race_wins
        GROUP BY name
        ORDER BY wins DESC
        LIMIT 5;
        ```

        Notice how the query uses the exact column name 'name' from the race_wins table schema, not 'driver' or 'winner'.

        Another example, if the user asks "Show me the number of races per year", your response should include:

        ```sql
        SELECT year, COUNT(DISTINCT venue) as num_races
        FROM race_results
        GROUP BY year
        ORDER BY year;
        ```

        Notice how this query uses the exact column names 'year' and 'venue' from the race_results table schema.

        For a question like "Tell me the driver with the longest racing career", your response should include:

        ```sql
        SELECT name, MIN(year) as first_year, MAX(year) as last_year, MAX(year) - MIN(year) as career_length
        FROM race_results
        GROUP BY name
        ORDER BY career_length DESC
        LIMIT 1;
        ```

        This query finds the driver with the longest span between their first and last race appearance.

        I will execute this query for you and provide the results. Make sure your response is clear and concise.

        IMPORTANT: Your response MUST include a SQL query in a code block with ```sql at the beginning and ``` at the end. If you don't include a SQL query, the system will not be able to execute it and provide results.
        """

        # Call the model with the formatted prompt
        import time
        start_time = time.time()
        response_obj = self.llm(prompt)
        execution_time = time.time() - start_time

        # Convert response to string if it's not already
        response_text = str(response_obj)

        # Extract SQL query if present in the response
        sql_query = None
        sql_result = None

        # Look for SQL code blocks in the response
        import re
        sql_matches = re.findall(r'```sql\s*([\s\S]*?)\s*```', response_text)

        if sql_matches:
            # Use the first SQL query found
            sql_query = sql_matches[0].strip()

            try:
                # Execute the SQL query
                logger.info(f"Executing SQL query: {sql_query}")
                with self.db_engine.connect() as conn:
                    result = conn.execute(text(sql_query))
                    rows = result.fetchall()
                    columns = result.keys()

                    # Convert to DataFrame and then to markdown table
                    df = pd.DataFrame(rows, columns=columns)
                    sql_result = df.to_markdown(index=False) if not df.empty else "No results found."

                    # Always add a direct answer at the beginning for clarity
                    if len(rows) > 0:
                        # For all queries, extract key information from the first row or summarize results
                        first_row = df.iloc[0].to_dict()

                        # Check if the response already starts with a clear answer
                        has_clear_answer = False
                        answer_patterns = ["**Answer:**", "**Results Summary:**", "The driver with", "The team with",
                                          "The most", "The top", "The longest", "The highest", "The best"]

                        for pattern in answer_patterns:
                            if pattern.lower() in response_text[:200].lower():
                                has_clear_answer = True
                                break

                        # If no clear answer is found, create one based on the query and results
                        if not has_clear_answer:
                            # For superlative queries, highlight the top result
                            if any(term in query.lower() for term in ["most", "top", "best", "highest", "longest", "greatest"]):
                                # Extract the key column names for a better answer
                                key_cols = [col for col in df.columns if col.lower() not in ['count', 'sum', 'avg', 'min', 'max', 'index']]
                                value_cols = [col for col in df.columns if col.lower() in ['count', 'wins', 'championships', 'points', 'races']]

                                if key_cols and value_cols:
                                    key_val = first_row[key_cols[0]]
                                    metric_val = first_row[value_cols[0]]
                                    metric_name = value_cols[0]
                                    answer = f"**Answer:** {key_val} has the most {metric_name} with {metric_val}.\n\n"
                                else:
                                    # Generic answer with the first row data
                                    answer = f"**Answer:** The top result is {first_row}.\n\n"
                            else:
                                # For other queries, provide a summary
                                answer = f"**Results Summary:** Found {len(rows)} records.\n\n"

                            # Add the answer to the beginning of the response
                            response_text = answer + response_text

                    # Add the SQL result to the response if not already present
                    if "Query Results" not in response_text and "SQL Query Result" not in response_text:
                        response_text += f"\n\n### Query Results\n\n{sql_result}"

            except Exception as e:
                # If there's an error executing the query, add it to the response
                error_message = f"Error executing SQL query: {str(e)}"
                logger.error(error_message)
                response_text += f"\n\n### Error Executing Query\n\n```\n{error_message}\n```"

        # Add the assistant's response to the conversation history
        self.messages.append({"role": "assistant", "content": response_text})

        return SQLQueryOutput(
            response=response_text,
            sql_query=sql_query,
            execution_time=execution_time
        )

    def run(self, query: str) -> str:
        """Legacy method for backward compatibility."""
        result = self.forward(inputs=SQLQueryInput(query=query))
        return result.response

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        semantic_model_str = json.dumps(self.semantic_model, indent=2)

        return dedent(f"""
        You are SQL Agent-X, an elite SQL Data Scientist specializing in:
        - Historical race analysis
        - Driver performance metrics
        - Team championship insights
        - Track statistics and records
        - Performance trend analysis
        - Race strategy evaluation

        You combine deep F1 knowledge with advanced SQL expertise to uncover insights from decades of racing data.
        You are dynamic and flexible, able to handle any type of question about the F1 data without relying on predefined patterns.
        You can generate SQL queries for any question, no matter how complex, by understanding the schema and relationships between tables.

        When a user messages you, determine if you need to query the database or can respond directly.

        If you can respond directly, do so.

        If you need to query the database to answer the user's question, follow these steps:

        1. First identify the tables you need to query from the semantic model.

        2. Then, ALWAYS use the `search_knowledge_base(table_name)` function to get table metadata, rules and sample queries.

        3. ALWAYS use the `describe_table(table_name)` function to get the actual schema and sample data from the database. This is critical to ensure you use the correct column names.

        4. If table rules are provided, ALWAYS follow them.

        5. Then, think step-by-step about query construction, don't rush this step.

        6. Follow a chain of thought approach before writing SQL, ask clarifying questions where needed.

        7. Then, using all the information available, create one single syntactically correct SQL query to accomplish your task.
           - Be creative and flexible in your query construction
           - Don't rely on predefined patterns - adapt to the specific question
           - Use appropriate SQL functions and operations based on the question
           - Consider different ways to interpret the question and choose the most appropriate
           - ALWAYS include a SQL query in your response - this is critical
           - NEVER stop after just retrieving schema information - always proceed to generate and execute a query

        8. If you need to join tables, check the `semantic_model` for the relationships between the tables.

        9. If you cannot find relevant tables, columns or relationships, stop and ask the user for more information.

        10. Once you have a syntactically correct query, run it using the `run_sql_query` function.

        11. When running a query:
        - Do not add a `;` at the end of the query.
        - Always provide a limit unless the user explicitly asks for all results.

        12. After you run the query, analyse the results and return the answer in markdown format.

        13. ALWAYS start your response with a direct answer to the user's question. For example:
           - "The driver with the most race wins is Michael Schumacher with 91 wins."
           - "Ferrari has won the most Constructor Championships with 16 titles."
           - "The driver with the longest career is Kimi Räikkönen who raced from 2001 to 2021."

        14. Always show the user the SQL you ran to get the answer.

        15. Continue till you have accomplished the task.

        16. Show results as a table or a chart if possible.

        After finishing your task, ask the user relevant followup questions like "was the result okay, would you like me to fix any problems?"

        If the user says yes, get the previous query using the `get_tool_call_history(num_calls=3)` function and fix the problems.

        If the user wants to see the SQL, get it using the `get_tool_call_history(num_calls=3)` function.

        Finally, here are the set of rules that you MUST follow:

        - Use the `search_knowledge_base(table_name)` function to get table information before writing a query.
        - Do not use phrases like "based on the information provided" or "from the knowledge base".
        - Always show the SQL queries you use to get the answer.
        - Make sure your query accounts for duplicate records.
        - Make sure your query accounts for null values.
        - If you run a query, explain why you ran it.
        - NEVER, EVER RUN CODE TO DELETE DATA OR ABUSE THE LOCAL SYSTEM
        - ALWAYS FOLLOW THE `table rules` if provided. NEVER IGNORE THEM.
        - Be dynamic and flexible - don't rely on hardcoded patterns for different question types.
        - Adapt your query approach based on the specific question being asked.
        - Always include the query results in your response.

        The `semantic_model` contains information about tables and the relationships between them.

        If the users asks about the tables you have access to, simply share the table names from the `semantic_model`.

        <semantic_model>
        {semantic_model_str}
        </semantic_model>

        You have the following functions available:

        1. search_knowledge_base(table_name: str) -> str
           - Get metadata, rules, and sample queries for a table

        2. describe_table(table_name: str) -> str
           - Get the schema of a table from the database

        3. run_sql_query(query: str) -> str
           - Run a SQL query and return the results

        4. get_tool_call_history(num_calls: int = 3) -> str
           - Get the history of recent tool calls
        """)

    def rename_session(self, new_name: str) -> None:
        """Rename the current session."""
        self.session_name = new_name


def get_sql_agent(model_name: str = "openai:gpt-4o") -> SQLAgent:
    """Get an instance of the SQL Agent.

    Args:
        model_name: The model to use for the agent

    Returns:
        An initialized SQL Agent
    """
    return SQLAgent(model_name=model_name)
