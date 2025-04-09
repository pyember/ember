"""NFT Assistant Runner.

This script demonstrates how to run the NFT Education Assistant.
"""

import os
import sys
from nft_education_assistant import NFTEducationAssistant

def main() -> None:
    """Run the NFT Education Assistant example."""
    # Check if OPENAI_API_KEY is set in the environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key using: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Check for GOOGLE_API_KEY environment variable
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable not set.")
        print("Please set your API key using: export GOOGLE_API_KEY='your-key-here'")
        print("This is required for Deepmind models.")
    
    # Check if Anthropic API key is provided as command line argument
    if len(sys.argv) > 1:
        os.environ["ANTHROPIC_API_KEY"] = sys.argv[1]
        print("Set ANTHROPIC_API_KEY from command line argument.")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY not set. Required for Anthropic models.")

    # Get model names from command line arguments or use recommended configuration
    ensemble_model = sys.argv[2] if len(sys.argv) > 2 else "deepmind:gemini-1.5-pro"
    judge_model = sys.argv[3] if len(sys.argv) > 3 else "openai:gpt-4o"

    print(f"Using ensemble model: {ensemble_model}")
    print(f"Using judge model: {judge_model}")

    # Create an instance of the NFT Education Assistant
    # Default to OpenAI model, but use Deepmind if specified
    model_name = "deepmind:gemini-1.5-pro" if "deepmind" in ensemble_model else "openai:gpt-4o"
    assistant = NFTEducationAssistant(model_name=model_name)

    # Example questions to demonstrate the assistant
    example_questions = [
        "What is an NFT?",
        "How do I create my first NFT?",
        "What is the environmental impact of NFTs?",
        "How can artists make money with NFTs?",
        "What are the most popular NFT marketplaces?"
    ]
    
    # User expertise levels for examples
    expertise_level = "beginner"
    
    # Run the example questions
    for i, question in enumerate(example_questions, 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {question}")
        
        # Process the question using explain_concept method
        response = assistant.explain_concept(question, expertise_level)
        
        # Display the results
        print("\nAnswer:")
        print(response)
        print("\n" + "-" * 80)
    
    # Interactive mode
    print("\n--- Interactive Mode ---")
    print("Type your questions about NFTs (or 'exit' to quit)")
    print("Default expertise level is 'beginner'. To change, type 'expertise:level'")
    
    current_expertise = "beginner"
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ("exit", "quit", "q"):
            break
        
        # Check if user wants to change expertise level
        if question.lower().startswith("expertise:"):
            try:
                current_expertise = question.split(":")[1].strip().lower()
                print(f"Expertise level set to: {current_expertise}")
                continue
            except IndexError:
                print("Invalid format. Use 'expertise:level' (e.g., expertise:intermediate)")
                continue
            
        response = assistant.explain_concept(question, current_expertise)
        print("\nAnswer:")
        print(response)

if __name__ == "__main__":
    main()
