from ember.api import models

class NFTEducationAssistant:
    def __init__(self, model_name="openai:gpt-4o"):
        """
        Initialize the NFT Education Assistant with a customizable model.

        Args:
            model_name (str, optional): Model to use for explanations. Default is "openai:gpt-4o".
                Options include:
                - OpenAI: "openai:gpt-4o", "openai:gpt-4o-mini", "openai:gpt-4", "openai:gpt-4-turbo", "openai:gpt-3.5-turbo"
                - Anthropic: "anthropic:claude-3-opus", "anthropic:claude-3-5-sonnet", "anthropic:claude-3-5-haiku", "anthropic:claude-3-7-sonnet"
                - Deepmind: "deepmind:gemini-1.5-pro", "deepmind:gemini-1.5-flash", "deepmind:gemini-2.0-pro"
        """
        # Note: API keys should be set as environment variables before running this code:
        # - OPENAI_API_KEY for OpenAI models
        # - ANTHROPIC_API_KEY for Anthropic models
        # - GOOGLE_API_KEY for Deepmind models

        # Create the model
        self.model = models.model(model_name, temperature=0.7)

    def explain_concept(self, query, user_expertise_level):
        """Provides personalized explanations of NFT/poker concepts

        Args:
            query (str): The NFT or poker concept to explain
            user_expertise_level (str): The user's expertise level (e.g., "beginner", "intermediate", "expert")

        Returns:
            str: The personalized explanation
        """
        # Format the prompt to include the expertise level
        prompt = f"""
        You are an NFT and Poker Education Assistant. Your task is to explain concepts related to NFTs and poker
        in a way that matches the user's expertise level.

        EXPERTISE LEVEL: {user_expertise_level}

        CONCEPT TO EXPLAIN: {query}

        Please provide a clear, accurate explanation of this concept that is appropriate for someone at the
        {user_expertise_level} level. Include relevant examples and analogies where helpful.
        """

        # Call the model with the formatted prompt
        response = self.model(prompt)

        # Return the response text
        return response

# Example usage
if __name__ == "__main__":
    assistant = NFTEducationAssistant()
    result = assistant.explain_concept(
        query="What is a non-fungible token?",
        user_expertise_level="beginner"
    )
    print(f"Explanation: {result}")
