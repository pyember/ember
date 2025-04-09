import os
from nft_education_assistant import NFTEducationAssistant

def main():
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Check if API keys are available
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set")
    if not anthropic_api_key:
        print("Warning: ANTHROPIC_API_KEY environment variable not set")

    # Create an instance of the NFT Education Assistant
    assistant = NFTEducationAssistant(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key
    )

    # Example queries at different expertise levels
    test_cases = [
        {
            "query": "What is a non-fungible token?",
            "expertise_level": "beginner"
        },
        {
            "query": "How do smart contracts work with NFTs?",
            "expertise_level": "intermediate"
        },
        {
            "query": "What are the implications of ERC-721 vs ERC-1155 for poker-based NFTs?",
            "expertise_level": "expert"
        }
    ]

    # Test each query
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Query: {test_case['query']}")
        print(f"Expertise Level: {test_case['expertise_level']}")

        # Get explanation
        result = assistant.explain_concept(
            query=test_case["query"],
            user_expertise_level=test_case["expertise_level"]
        )

        # Print result
        print("\nResult:")
        if isinstance(result, dict) and "synthesized_response" in result:
            print(result["synthesized_response"])
        else:
            print(result)

if __name__ == "__main__":
    main()
