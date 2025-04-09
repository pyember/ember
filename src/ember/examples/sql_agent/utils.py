"""SQL Agent Utilities Module.

This module provides utility functions for the SQL Agent.
"""

import json
import logging
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_json(myjson: Any) -> bool:
    """Check if a string is valid JSON.
    
    Args:
        myjson: The string to check.
        
    Returns:
        bool: True if the string is valid JSON, False otherwise.
    """
    try:
        json.loads(myjson)
        return True
    except (ValueError, TypeError):
        return False

def add_message(
    messages: List[Dict[str, Any]], 
    role: str, 
    content: str, 
    tool_calls: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Safely add a message to the messages list.
    
    Args:
        messages: The list of messages to add to.
        role: The role of the message sender (user, assistant, system).
        content: The content of the message.
        tool_calls: Optional list of tool calls.
        
    Returns:
        The updated list of messages.
    """
    if messages is None:
        messages = []

    messages.append({
        "role": role,
        "content": content,
        "tool_calls": tool_calls
    })

    return messages

def export_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Export chat history as markdown.
    
    Args:
        messages: The list of messages to export.
        
    Returns:
        str: The chat history formatted as markdown.
    """
    if not messages:
        return "# SQL Agent - Chat History\n\nNo messages in history."

    chat_text = "# SQL Agent - Chat History\n\n"

    for msg in messages:
        role = "🤖 Assistant" if msg["role"] == "assistant" else "👤 User"
        chat_text += f"### {role}\n{msg['content']}\n\n"

    return chat_text

def display_tool_calls(tool_calls_container: Any, tools: List[Dict[str, Any]]) -> None:
    """Display tool calls in a streamlit container with expandable sections.
    
    Args:
        tool_calls_container: The streamlit container to display in.
        tools: The list of tools to display.
    """
    try:
        if not tools:
            return

        with tool_calls_container.container():
            for tool_call in tools:
                tool_name = tool_call.get("tool_name", "Unknown Tool")
                tool_args = tool_call.get("tool_args", {})
                content = tool_call.get("content", None)

                with tool_calls_container.expander(
                    f"🛠️ {tool_name.replace('_', ' ').title()}",
                    expanded=False,
                ):
                    # Show query with syntax highlighting
                    if isinstance(tool_args, dict) and "query" in tool_args:
                        tool_calls_container.code(tool_args["query"], language="sql")

                    # Display arguments in a more readable format
                    if tool_args and tool_args != {"query": None}:
                        tool_calls_container.markdown("**Arguments:**")
                        tool_calls_container.json(tool_args)

                    if content is not None:
                        try:
                            if is_json(content):
                                try:
                                    parsed_content = json.loads(content)
                                    tool_calls_container.markdown("**Results:**")
                                    tool_calls_container.json(parsed_content)
                                except:
                                    tool_calls_container.markdown("**Results:**")
                                    tool_calls_container.markdown(content)
                            else:
                                tool_calls_container.markdown("**Results:**")
                                tool_calls_container.markdown(content)
                        except Exception as e:
                            logger.debug(f"Skipped tool call content: {e}")

    except Exception as e:
        logger.error(f"Error displaying tool calls: {str(e)}")
        tool_calls_container.error("Failed to display tool results")

# Custom CSS for the Streamlit app
CUSTOM_CSS = """
<style>
/* Main Styles */
.main-title {
    text-align: center;
    background: linear-gradient(45deg, #FF4B2B, #FF416C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    padding: 1em 0;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 2em;
}

.stButton button {
    width: 100%;
    border-radius: 20px;
    margin: 0.2em 0;
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.chat-container {
    border-radius: 15px;
    padding: 1em;
    margin: 1em 0;
    background-color: #f5f5f5;
}

.sql-result {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1em;
    margin: 1em 0;
    border-left: 4px solid #FF4B2B;
}

.status-message {
    padding: 1em;
    border-radius: 10px;
    margin: 1em 0;
}

.success-message {
    background-color: #d4edda;
    color: #155724;
}

.error-message {
    background-color: #f8d7da;
    color: #721c24;
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
    .chat-container {
        background-color: #2b2b2b;
    }

    .sql-result {
        background-color: #1e1e1e;
    }
}
</style>
""" 