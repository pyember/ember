"""SQL Agent Streamlit Application.

This module provides a web interface for the SQL Agent using Streamlit.
"""

import os
import streamlit as st
import pandas as pd
import logging
import nest_asyncio
from pathlib import Path

from ember.examples.sql_agent.sql_agent import SQLAgent
from ember.examples.sql_agent.utils import CUSTOM_CSS, add_message, display_tool_calls, export_chat_history
from ember.examples.sql_agent.load_f1_data import load_f1_data

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

        if st.sidebar.button("🚀 Load F1 Data"):
            with st.spinner("🔄 Loading data into database..."):
                load_f1_data()
            st.success("✅ Data loaded successfully!")

def about_widget() -> None:
    """Display an about section in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This SQL Assistant helps you analyze Formula 1 data from 1950 to 2020 using natural language queries.

    Built with:
    - 🚀 Ember
    - 💫 Streamlit
    - 📊 SQLAlchemy
    """)

def select_model_widget() -> str:
    """Display a model selection widget in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Selection")
    
    model_options = {
        "OpenAI GPT-4o": "openai:gpt-4o",
        "OpenAI GPT-4 Turbo": "openai:gpt-4-turbo",
        "OpenAI GPT-3.5 Turbo": "openai:gpt-3.5-turbo",
        "Anthropic Claude 3 Opus": "anthropic:claude-3-opus",
        "Anthropic Claude 3 Sonnet": "anthropic:claude-3-sonnet",
        "DeepMind Gemini 1.5 Pro": "deepmind:gemini-1.5-pro",
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Select LLM Model",
        options=list(model_options.keys()),
        index=0,
        key="model_selection"
    )
    
    # Check for API keys
    model_provider = selected_model_name.split()[0].lower()
    api_key_env = f"{model_provider.upper()}_API_KEY"
    
    if api_key_env not in os.environ or not os.environ[api_key_env]:
        st.sidebar.warning(f"⚠️ {api_key_env} not set. Set this environment variable before using {selected_model_name}.")
    
    return model_options[selected_model_name]

def main() -> None:
    """Main function to run the Streamlit app."""
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
    # API key check
    ####################################################################
    if "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ and "DEEPMIND_API_KEY" not in os.environ:
        st.warning("""
        ⚠️ No API keys found in environment variables. 
        
        You need to set at least one of the following environment variables:
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY
        - DEEPMIND_API_KEY
        
        Example:
        ```
        export OPENAI_API_KEY='your_api_key_here'
        ```
        """)

    ####################################################################
    # Auto-load data if not already loaded
    ####################################################################
    if "data_loaded" not in st.session_state or not st.session_state["data_loaded"]:
        with st.spinner("🔄 Loading F1 data..."):
            try:
                db_path = "f1_data.db"
                # Only load data if database doesn't exist or is empty
                if not os.path.exists(db_path) or os.path.getsize(db_path) < 1000:
                    load_f1_data(db_path)
                st.session_state["data_loaded"] = True
                st.success("✅ F1 data ready!")
            except Exception as e:
                st.error(f"❌ Error loading F1 data: {str(e)}")
                st.session_state["data_loaded"] = False

    ####################################################################
    # Sidebar widgets
    ####################################################################
    sidebar_widget()
    selected_model = select_model_widget()
    about_widget()

    ####################################################################
    # Initialize SQL Agent
    ####################################################################
    if "sql_agent" not in st.session_state or st.session_state["sql_agent"] is None:
        try:
            st.session_state["sql_agent"] = SQLAgent(
                database_url="sqlite:///f1_data.db",
                model_name=selected_model,
                temperature=0.0
            )
        except Exception as e:
            st.error(f"❌ Error initializing SQL Agent: {str(e)}")
            return

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    ####################################################################
    # Display chat history
    ####################################################################
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                # Display tool calls if available
                if message.get("tool_calls"):
                    tool_calls_container = st.container()
                    display_tool_calls(tool_calls_container, message["tool_calls"])

    ####################################################################
    # Chat input
    ####################################################################
    if prompt := st.chat_input("Ask a question about Formula 1 data..."):
        # Add user message to chat history
        st.session_state["messages"] = add_message(
            st.session_state["messages"], "user", prompt
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from SQL Agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Process the query
                    chat_response = st.session_state["sql_agent"].chat(
                        prompt, st.session_state["messages"]
                    )
                    
                    # Update messages with assistant's response
                    st.session_state["messages"] = chat_response["messages"]
                    
                    # Display the answer
                    st.markdown(chat_response["response"]["answer"])
                    
                    # Display tool calls
                    tool_calls_container = st.container()
                    
                    # Create tool calls for display
                    tool_calls = [
                        {
                            "tool_name": "generate_sql_query",
                            "tool_args": {"question": prompt},
                            "content": chat_response["response"]["sql_query"]
                        },
                        {
                            "tool_name": "execute_sql_query",
                            "tool_args": {"query": chat_response["response"]["sql_query"]},
                            "content": str(chat_response["response"]["query_result"])
                        }
                    ]
                    
                    display_tool_calls(tool_calls_container, tool_calls)
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    logger.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main() 