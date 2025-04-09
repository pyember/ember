import nest_asyncio
import streamlit as st
import logging
import os
import pandas as pd
import re
from sqlalchemy import text

from agent import get_sql_agent
from utils import CUSTOM_CSS, add_message, display_tool_calls, export_chat_history
from load_f1_data import load_f1_data
from load_knowledge import load_knowledge

# Set the OpenAI API key from environment variable
# Make sure to set your OPENAI_API_KEY environment variable before running this app
# Example: export OPENAI_API_KEY="your-api-key-here"
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set. Some features may not work properly.")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up Streamlit page
st.set_page_config(
    page_title="SQL Agent with Ember",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS with dark mode support
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def sidebar_widget() -> None:
    """Display a sidebar with sample user queries"""
    with st.sidebar:
        st.markdown("#### 🏆 Sample Queries")

        if st.button("📋 Show Tables"):
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"] = add_message(
                st.session_state["messages"],
                "user",
                "Which tables do you have access to?"
            )

        if st.button("🥇 Most Race Wins"):
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"] = add_message(
                st.session_state["messages"],
                "user",
                "Which driver has the most race wins?"
            )

        if st.button("🏆 Constructor Champs"):
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"] = add_message(
                st.session_state["messages"],
                "user",
                "Which team won the most Constructors Championships?"
            )

        if st.button("⏳ Longest Career"):
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"] = add_message(
                st.session_state["messages"],
                "user",
                "Tell me the name of the driver with the longest racing career? Also tell me when they started and when they retired."
            )

        if st.button("📈 Races per Year"):
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"] = add_message(
                st.session_state["messages"],
                "user",
                "Show me the number of races per year."
            )

        if st.button("🔍 Team Performance"):
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"] = add_message(
                st.session_state["messages"],
                "user",
                "Write a query to identify the drivers that won the most races per year from 2010 onwards and the position of their team that year."
            )

        st.markdown("---")

        st.markdown("#### 🛠️ Utilities")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 New Chat"):
                st.session_state["sql_agent"] = None
                st.session_state["messages"] = []
                st.rerun()

        with col2:
            fn = "sql_agent_chat_history.md"

            if st.download_button(
                "💾 Export Chat",
                export_chat_history(st.session_state.get("messages", [])),
                file_name=fn,
                mime="text/markdown",
            ):
                st.sidebar.success("Chat history exported!")

        if st.sidebar.button("🚀 Load Data & Knowledge"):
            with st.spinner("🔄 Loading data into database..."):
                load_f1_data()
            with st.spinner("📚 Loading knowledge base..."):
                load_knowledge()
            st.success("✅ Data and knowledge loaded successfully!")

def about_widget() -> None:
    """Display an about section in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This SQL Assistant helps you analyze Formula 1 data from 1950 to 2020 using natural language queries.

    Built with:
    - 🚀 Ember
    - 💫 Streamlit
    """)

def rename_session_widget(agent) -> None:
    """Rename the current session of the agent"""
    container = st.sidebar.container()
    session_row = container.columns([3, 1], vertical_alignment="center")

    # Initialize session_edit_mode if needed
    if "session_edit_mode" not in st.session_state:
        st.session_state.session_edit_mode = False

    with session_row[0]:
        if st.session_state.session_edit_mode:
            new_session_name = st.text_input(
                "Session Name",
                value=agent.session_name,
                key="session_name_input",
                label_visibility="collapsed",
            )
        else:
            st.markdown(f"Session Name: **{agent.session_name}**")

    with session_row[1]:
        if st.session_state.session_edit_mode:
            if st.button("✓", key="save_session_name", type="primary"):
                if new_session_name:
                    agent.rename_session(new_session_name)
                    st.session_state.session_edit_mode = False
                    container.success("Renamed!")
        else:
            if st.button("✎", key="edit_session_name"):
                st.session_state.session_edit_mode = True

def main() -> None:
    ####################################################################
    # App header
    ####################################################################
    st.markdown(
        "<h1 class='main-title'>SQL Agent with Ember</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Your intelligent SQL Agent that can think, analyze and reason, powered by Ember</p>",
        unsafe_allow_html=True,
    )

    ####################################################################
    # Auto-load data and knowledge if not already loaded
    ####################################################################
    if "data_loaded" not in st.session_state or not st.session_state["data_loaded"]:
        with st.spinner("🔄 Loading data into database..."):
            try:
                load_f1_data()
                st.session_state["data_loaded"] = True
                st.success("✅ F1 data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading F1 data: {str(e)}")
                st.session_state["data_loaded"] = False

        with st.spinner("📚 Loading knowledge base..."):
            try:
                load_knowledge()
                st.session_state["knowledge_loaded"] = True
                st.success("✅ Knowledge base loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading knowledge base: {str(e)}")
                st.session_state["knowledge_loaded"] = False

    ####################################################################
    # Model selector
    ####################################################################
    model_options = {
        "GPT-4o": "openai:gpt-4o",
        "GPT-4o-mini": "openai:gpt-4o-mini",
        "GPT-4-turbo": "openai:gpt-4-turbo",
        "GPT-3.5-turbo": "openai:gpt-3.5-turbo",
    }

    selected_model = st.sidebar.selectbox(
        "Select a model",
        options=list(model_options.keys()),
        index=0,
        key="model_selector",
    )

    model_id = model_options[selected_model]

    ####################################################################
    # Initialize Agent
    ####################################################################
    sql_agent = None

    if (
        "sql_agent" not in st.session_state
        or st.session_state["sql_agent"] is None
        or st.session_state.get("current_model") != model_id
    ):
        logger.info("---*--- Creating new SQL agent ---*---")
        sql_agent = get_sql_agent(model_name=model_id)
        st.session_state["sql_agent"] = sql_agent
        st.session_state["current_model"] = model_id
    else:
        sql_agent = st.session_state["sql_agent"]

    ####################################################################
    # Initialize messages if not already done
    ####################################################################
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    ####################################################################
    # Sidebar
    ####################################################################
    sidebar_widget()

    ####################################################################
    # Get user input
    ####################################################################
    if prompt := st.chat_input("👋 Ask me about F1 data from 1950 to 2020!"):
        st.session_state["messages"] = add_message(st.session_state["messages"], "user", prompt)

    ####################################################################
    # Display chat history
    ####################################################################
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            content = message["content"]
            if content is not None:
                with st.chat_message(message["role"]):
                    # Display tool calls if they exist in the message
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(content)

    ####################################################################
    # Generate response for user message
    ####################################################################
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )

    if last_message and last_message.get("role") == "user":
        question = last_message["content"]

        with st.chat_message("assistant"):
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()

            with st.spinner("🤔 Thinking..."):
                try:
                    # Run the agent
                    from agent import SQLQueryInput
                    result = sql_agent.forward(inputs=SQLQueryInput(query=question))

                    # Display tool calls if available
                    if hasattr(sql_agent, 'tool_calls') and sql_agent.tool_calls:
                        display_tool_calls(tool_calls_container, sql_agent.tool_calls)

                    # Extract SQL query and results if present
                    response_text = result.response
                    sql_query = result.sql_query

                    # Check if we need to extract and execute a query from the response
                    import re

                    # First, check if there's a describe_table call in the response
                    describe_matches = re.findall(r'describe_table\("([^"]+)"\)', response_text)

                    if describe_matches:
                        # Execute the describe_table function for each match
                        for table_name in describe_matches:
                            try:
                                # Call the describe_table function
                                table_info = sql_agent.describe_table(table_name)

                                # Add the table info to the response
                                response_text += f"\n\n{table_info}\n\n"

                                # Let the LLM generate the appropriate query in its response
                                # We don't need hardcoded queries anymore as the agent is now more dynamic

                            except Exception as e:
                                error_msg = f"Error executing describe_table for {table_name}: {str(e)}"
                                response_text += f"\n\n{error_msg}\n\n"
                                st.error(error_msg)

                    # Display the full response
                    resp_container.markdown(response_text)

                    # If there's a SQL query in the response, execute it
                    sql_matches = re.findall(r'```sql\s*([\s\S]*?)\s*```', response_text)
                    if sql_matches and not describe_matches:  # Only if we haven't already executed a query above
                        sql_query = sql_matches[0].strip()
                        try:
                            # Execute the SQL query directly
                            with sql_agent.db_engine.connect() as conn:
                                df = pd.read_sql(sql_query, conn)

                                # Display the results in a dataframe for better visualization
                                st.markdown("### Interactive Results")
                                st.dataframe(df, use_container_width=True)

                                # If the response doesn't already contain the results, add them
                                if "Query Results" not in response_text and "SQL Query Result" not in response_text:
                                    # Update the response container with the new content
                                    updated_response = response_text + f"\n\n### Query Results\n\n{df.to_markdown(index=False)}"
                                    resp_container.markdown(updated_response)
                                    # Update the response text for the session state
                                    response_text = updated_response
                        except Exception as e:
                            st.error(f"Error executing SQL query: {str(e)}")
                    elif not sql_matches and describe_matches:  # If we have describe_table calls but no SQL query
                        # The agent retrieved schema information but didn't generate a SQL query
                        # Let's generate a simple query based on the table and question
                        note = "\n\n**Note: The agent retrieved schema information but didn't generate a SQL query. Generating a simple query based on the table...**"
                        response_text += note
                        resp_container.markdown(response_text)

                        # Generate a simple query based on the table name and question
                        table_name = describe_matches[0]  # Use the first table mentioned

                        # Dynamically explore and map the database schema
                        db_schema_info = ""
                        all_tables = []
                        table_columns = {}
                        relationships = []

                        try:
                            with sql_agent.db_engine.connect() as conn:
                                # Get all tables in the database
                                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                                all_tables = [row[0] for row in result.fetchall()]

                                # For each table, get its columns
                                for table in all_tables:
                                    result = conn.execute(text(f"PRAGMA table_info({table})"))
                                    columns = result.fetchall()
                                    table_columns[table] = [(col[1], col[2]) for col in columns]

                                    # Get sample data for each table
                                    try:
                                        sample = conn.execute(text(f"SELECT * FROM {table} LIMIT 3")).fetchall()
                                        if sample:
                                            # Add sample data to the schema info
                                            table_columns[table].append(("sample_data", sample))
                                    except:
                                        pass

                                # Try to infer relationships between tables
                                for table1 in all_tables:
                                    for table2 in all_tables:
                                        if table1 != table2:
                                            # Get column names for both tables
                                            cols1 = [col[0] for col in table_columns[table1] if isinstance(col, tuple) and len(col) == 2]
                                            cols2 = [col[0] for col in table_columns[table2] if isinstance(col, tuple) and len(col) == 2]

                                            # Look for common column names that might indicate relationships
                                            for col1 in cols1:
                                                if col1 in cols2 or f"{table1[:-1]}_id" == col1 or f"{table2[:-1]}_id" == col1:
                                                    relationships.append((table1, table2, col1))
                        except Exception as e:
                            logger.error(f"Error exploring database schema: {str(e)}")

                        # Format the schema information
                        db_schema_info = "Database Schema:\n"
                        for table in all_tables:
                            db_schema_info += f"\nTable: {table}\n"
                            db_schema_info += "Columns:\n"
                            for col in table_columns[table]:
                                if isinstance(col, tuple) and len(col) == 2 and col[0] != "sample_data":
                                    db_schema_info += f"- {col[0]}: {col[1]}\n"

                        if relationships:
                            db_schema_info += "\nPossible Relationships:\n"
                            for rel in relationships:
                                db_schema_info += f"- {rel[0]} may be related to {rel[1]} through column {rel[2]}\n"

                        # Get specific information about the table mentioned in the question
                        table_info = ""
                        if table_name in table_columns:
                            table_info = f"\nFocused Table: {table_name}\n"
                            table_info += "Columns:\n"
                            for col in table_columns[table_name]:
                                if isinstance(col, tuple) and len(col) == 2 and col[0] != "sample_data":
                                    table_info += f"- {col[0]}: {col[1]}\n"

                            # Add sample data if available
                            for col in table_columns[table_name]:
                                if isinstance(col, tuple) and len(col) == 2 and col[0] == "sample_data":
                                    table_info += "\nSample Data:\n"
                                    for row in col[1][:3]:  # Show up to 3 rows
                                        table_info += f"{row}\n"

                        # Send a follow-up request to the LLM to generate a SQL query
                        follow_up_prompt = f"""Based on the following database schema, generate a SQL query to answer: {question}

                        {db_schema_info}

                        {table_info}

                        IMPORTANT INSTRUCTIONS:
                        1. Use ONLY the exact table names and column names listed above. Do not use placeholder names like 'table_name' or 'column1'.
                        2. For the current question, you should use the actual table name '{table_name}' in your query, not a placeholder.
                        3. Consider the relationships between tables if they are relevant to the question.
                        4. For questions about time spans or careers, use appropriate SQL functions like MIN(), MAX(), etc.
                        5. Be creative and flexible in your approach, but ensure the query will execute correctly.
                        6. Format your response as a SQL query inside a code block with ```sql at the beginning and ``` at the end.
                        7. Double-check that all table and column names in your query exactly match those in the schema.
                        """

                        # Call the LLM to generate a SQL query
                        try:
                            # Use the same model as the agent
                            llm = sql_agent.llm
                            response = llm(follow_up_prompt)

                            # Extract SQL query from the response
                            import re
                            sql_matches = re.findall(r'```sql\s*([\s\S]*?)\s*```', str(response))

                            if sql_matches:
                                simple_query = sql_matches[0].strip()

                                # Validate that the query only uses existing tables and columns
                                valid_columns = []
                                valid_tables = list(table_columns.keys())

                                for table in table_columns:
                                    for col in table_columns[table]:
                                        if isinstance(col, tuple) and len(col) == 2 and col[0] != "sample_data":
                                            valid_columns.append(col[0].lower())

                                # Add SQL keywords and functions to the valid list
                                sql_keywords = ['select', 'from', 'where', 'group', 'by', 'order', 'limit', 'as', 'min', 'max',
                                               'count', 'sum', 'avg', 'and', 'or', 'not', 'distinct', 'having', 'desc', 'asc',
                                               'join', 'inner', 'outer', 'left', 'right', 'on', 'case', 'when', 'then', 'else', 'end',
                                               'in', 'between', 'like', 'is', 'null', 'cast', 'coalesce', 'nullif', 'ifnull',
                                               'date', 'datetime', 'time', 'strftime', 'julianday', 'unixepoch', 'localtime',
                                               'year', 'month', 'day', 'hour', 'minute', 'second']

                                # Check if the query contains the table name
                                if table_name.lower() not in simple_query.lower():
                                    logger.warning(f"Query does not contain the table name: {table_name}")
                                    # Replace 'table_name' with the actual table name
                                    simple_query = simple_query.replace('table_name', table_name)
                                    # Also try to replace 'tablename' with the actual table name
                                    simple_query = simple_query.replace('tablename', table_name)

                                # Simple validation to catch obvious errors
                                invalid_columns = []
                                for col_name in re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', simple_query):
                                    if (col_name.lower() not in valid_columns and
                                        col_name.lower() not in valid_tables and
                                        col_name.lower() not in sql_keywords and
                                        not col_name.lower().startswith('sqlite_') and
                                        not col_name.isdigit()):
                                        invalid_columns.append(col_name)

                                if invalid_columns:
                                    # If invalid columns found, try to fix the query
                                    logger.warning(f"Invalid column names in query: {invalid_columns}")

                                    # Try to generate a corrected query
                                    correction_prompt = f"""The following SQL query contains invalid column or table names: {invalid_columns}

                                    Query: {simple_query}

                                    Valid table names are: {valid_tables}
                                    Valid column names are: {valid_columns}

                                    For this specific question, you should use the table '{table_name}' and its columns.

                                    Please correct the query to use only valid table names, column names, and SQL keywords.
                                    Do not use placeholder names like 'table_name' or 'column1' - use the actual names from the schema.
                                    Format your response as a SQL query inside a code block with ```sql at the beginning and ``` at the end.
                                    """

                                    try:
                                        correction_response = llm(correction_prompt)
                                        correction_matches = re.findall(r'```sql\s*([\s\S]*?)\s*```', str(correction_response))

                                        if correction_matches:
                                            simple_query = correction_matches[0].strip()
                                            logger.info(f"Corrected query: {simple_query}")
                                        else:
                                            # If no corrected query found, use a generic query
                                            simple_query = f"SELECT * FROM {table_name} LIMIT 10"
                                    except Exception as e:
                                        logger.error(f"Error correcting query: {str(e)}")
                                        simple_query = f"SELECT * FROM {table_name} LIMIT 10"
                            else:
                                # If no SQL query found, use a generic query
                                simple_query = f"SELECT * FROM {table_name} LIMIT 10"
                        except Exception as e:
                            # If there's an error, use a generic query
                            logger.error(f"Error generating SQL query: {str(e)}")
                            simple_query = f"SELECT * FROM {table_name} LIMIT 10"

                        try:
                            # Execute the simple query
                            import time
                            start_time = time.time()
                            with sql_agent.db_engine.connect() as conn:
                                df = pd.read_sql(simple_query, conn)
                            execution_time = time.time() - start_time

                            # Add the query and results to the response
                            response_text += f"\n\n```sql\n{simple_query}\n```\n\n### Query Results\n\n{df.to_markdown(index=False)}"

                            # Generate a clear, definitive answer based on the query results
                            # Instead of hardcoded patterns, use the LLM to generate a dynamic answer
                            if not df.empty:
                                # Convert the dataframe to a string representation
                                df_str = df.to_string(index=False)

                                # Create a prompt for the LLM to generate an answer
                                answer_prompt = f"""Based on the following SQL query and its results, provide a clear, direct answer to the question: '{question}'

                                SQL Query:
                                {simple_query}

                                Query Results:
                                {df_str}

                                Database Schema Summary:
                                {db_schema_info}

                                IMPORTANT INSTRUCTIONS:
                                1. Give a concise, definitive answer that directly addresses the question.
                                2. Start with 'Answer:' and focus on the key insights from the data.
                                3. Be specific and include actual numbers, names, and values from the query results.
                                4. If the question asks for a specific piece of information (like who has the most wins or longest career), clearly state that information.
                                5. If the query results show multiple records, summarize the most important findings.
                                6. Make your answer complete enough that someone could understand it without seeing the query or results.
                                7. Do not say things like 'Based on the query results' or 'According to the data' - just state the facts directly.
                                8. If the results are empty or don't answer the question, say so clearly.
                                """

                                try:
                                    # Use the same model as the agent
                                    llm = sql_agent.llm
                                    answer_response = llm(answer_prompt)

                                    # Extract the answer
                                    answer_text = str(answer_response).strip()

                                    # If the answer doesn't start with "Answer:", add it
                                    if not answer_text.startswith("Answer:"):
                                        answer_text = "Answer: " + answer_text

                                    # Format the answer
                                    direct_answer = f"\n\n**{answer_text}**"
                                    response_text = direct_answer + response_text
                                except Exception as e:
                                    # If there's an error, create a generic answer
                                    logger.error(f"Error generating answer: {str(e)}")

                                    # Create a generic answer based on the first row
                                    if len(df) > 0:
                                        # Extract key columns for a meaningful answer
                                        key_cols = [col for col in df.columns if col.lower() not in ['index', 'id']]
                                        if key_cols:
                                            # Create a summary of the first row
                                            first_row_summary = ", ".join([f"{col}: {df.iloc[0][col]}" for col in key_cols[:3]])
                                            direct_answer = f"\n\n**Answer: Based on the data, {first_row_summary}. See the full results below for more details.**"
                                        else:
                                            direct_answer = f"\n\n**Answer: Found {len(df)} records matching your query. See the full results below.**"
                                        response_text = direct_answer + response_text
                            else:
                                direct_answer = "\n\n**Answer: No records found matching your query.**"
                                response_text = direct_answer + response_text

                            resp_container.markdown(response_text)

                            # Display the results in a dataframe
                            st.markdown("### Interactive Results")
                            st.dataframe(df, use_container_width=True)

                            # Set the SQL query for display in the expander
                            sql_query = simple_query

                        except Exception as e:
                            error_msg = f"Error executing simple query: {str(e)}"
                            response_text += f"\n\n{error_msg}"
                            resp_container.markdown(response_text)
                            st.error(error_msg)

                    # Display SQL query if available (in a collapsible section)
                    if sql_query:
                        with st.expander("View SQL Query"):
                            st.code(sql_query, language="sql")

                    # Display execution time if available (as a small note)
                    try:
                        if hasattr(result, 'execution_time') and result.execution_time:
                            st.caption(f"Execution time: {result.execution_time:.2f} seconds")
                        elif 'execution_time' in locals():
                            st.caption(f"Execution time: {execution_time:.2f} seconds")
                    except Exception as e:
                        logger.warning(f"Could not display execution time: {str(e)}")

                    # Add the response to the messages
                    tool_calls_to_add = sql_agent.tool_calls if hasattr(sql_agent, 'tool_calls') else None
                    st.session_state["messages"] = add_message(
                        st.session_state["messages"],
                        "assistant",
                        result.response,
                        tool_calls_to_add
                    )

                except Exception as e:
                    logger.exception(e)
                    error_message = f"Sorry, I encountered an error: {str(e)}"

                    # Display error message
                    resp_container.error(error_message)

                    # Add error message to conversation history
                    st.session_state["messages"] = add_message(
                        st.session_state["messages"],
                        "assistant",
                        error_message
                    )

    ####################################################################
    # Rename session widget
    ####################################################################
    rename_session_widget(sql_agent)

    ####################################################################
    # About section
    ####################################################################
    about_widget()

if __name__ == "__main__":
    main()
