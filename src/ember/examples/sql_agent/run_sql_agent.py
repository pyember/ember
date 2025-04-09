"""SQL Agent Example Runner.

This script demonstrates how to use the SQL Agent to query a Formula 1 database.
"""

import os
import sys
from ember.examples.sql_agent.sql_agent import SQLAgent
from ember.examples.sql_agent.load_f1_data import load_f1_data

def main() -> None:
    """Run the SQL Agent example."""
    # Check if OPENAI_API_KEY is set in the environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key using: export OPENAI_API_KEY='your-key-here'")
    
    # Check if a database path is provided
    database_path = sys.argv[1] if len(sys.argv) > 1 else "f1_data.db"
    database_url = f"sqlite:///{database_path}"
    
    # Check if the database exists; if not, create it
    if not os.path.exists(database_path):
        print(f"Database {database_path} not found. Creating and loading data...")
        load_f1_data(database_path)
    
    # Get model name from command line arguments or use default
    model_name = sys.argv[2] if len(sys.argv) > 2 else "openai:gpt-4o"
    
    print(f"Using database: {database_path}")
    print(f"Using model: {model_name}")
    
    # Create an instance of the SQL Agent
    agent = SQLAgent(
        database_url=database_url,
        model_name=model_name,
        temperature=0.0
    )
    
    # Example questions to demonstrate the agent
    example_questions = [
        "Who won the most races in the 2019 season?",
        "Which team had the most points in the constructors championship in 2018?",
        "Who had the fastest lap in Monaco in 2019?",
        "Compare the performance of Hamilton and Vettel in 2018"
    ]
    
    # Run the example questions
    for i, question in enumerate(example_questions, 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {question}")
        
        # Process the question
        result = agent.query(question)
        
        # Display the results
        print("\nGenerated SQL Query:")
        print(result["sql_query"])
        
        print("\nQuery Results:")
        if result["query_result"]["success"]:
            if result["query_result"]["record_count"] > 0:
                import pandas as pd
                df = pd.DataFrame(result["query_result"]["results"])
                print(df.to_string(index=False))
            else:
                print("No results found.")
        else:
            print(f"Query failed: {result['query_result']['error']}")
        
        print("\nAnswer:")
        print(result["answer"])
        print("\n" + "-" * 80)
    
    # Interactive mode
    print("\n--- Interactive Mode ---")
    print("Type your questions about Formula 1 data (or 'exit' to quit)")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ("exit", "quit", "q"):
            break
            
        result = agent.query(question)
        
        print("\nSQL Query:")
        print(result["sql_query"])
        
        print("\nAnswer:")
        print(result["answer"])

if __name__ == "__main__":
    main() 