"""Basic usage examples for Ember-DSPy integration."""

import dspy

from ember.integrations.dspy import EmberLM


def example_basic_prediction():
    """Basic text classification with EmberLM."""
    print("=== Basic Prediction Example ===")

    # Initialize Ember backend with Claude
    ember_lm = EmberLM(model="claude-3-haiku-20240307", temperature=0.3)
    dspy.configure(lm=ember_lm)

    # Create a simple classifier
    classify = dspy.Predict("text -> sentiment")

    # Test texts
    texts = [
        "I absolutely love this new framework!",
        "This is terrible and doesn't work at all.",
        "The documentation could be better, but overall it's decent.",
    ]

    for text in texts:
        result = classify(text=text)
        print(f"Text: {text}")
        print(f"Sentiment: {result.sentiment}\n")


def example_chain_of_thought():
    """Chain of thought reasoning with EmberLM."""
    print("=== Chain of Thought Example ===")

    # Use GPT-4 for complex reasoning
    ember_lm = EmberLM(model="gpt-4", temperature=0.1)
    dspy.configure(lm=ember_lm)

    # Create chain of thought module
    solve = dspy.ChainOfThought("question -> answer")

    questions = [
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    ]

    for question in questions:
        result = solve(question=question)
        print(f"Question: {question}")
        print(f"Answer: {result.answer}")
        print(f"Reasoning: {result.rationale}\n")


def example_multi_hop_qa():
    """Multi-hop question answering with EmberLM."""
    print("=== Multi-Hop QA Example ===")

    # Use Claude for multi-hop reasoning
    ember_lm = EmberLM(model="claude-3-opus-20240229", temperature=0.2)
    dspy.configure(lm=ember_lm)

    class MultiHopQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_query = dspy.ChainOfThought("context, question -> search_query")
            self.generate_answer = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question, context=""):
            # First hop: generate search query
            query_result = self.generate_query(context=context, question=question)

            # Simulate retrieval (in real usage, you'd search a database)
            retrieved_context = (
                f"Retrieved info based on '{query_result.search_query}': [simulated context]"
            )

            # Second hop: generate answer with retrieved context
            full_context = f"{context}\n{retrieved_context}"
            answer_result = self.generate_answer(context=full_context, question=question)

            return dspy.Prediction(
                answer=answer_result.answer,
                search_query=query_result.search_query,
                rationale=answer_result.rationale,
            )

    qa = MultiHopQA()
    result = qa(question="What year did the company that acquired WhatsApp go public?")

    print("Question: What year did the company that acquired WhatsApp go public?")
    print(f"Generated Query: {result.search_query}")
    print(f"Answer: {result.answer}")
    print(f"Rationale: {result.rationale}\n")


def example_model_switching():
    """Demonstrate switching between models dynamically."""
    print("=== Model Switching Example ===")

    # Define task
    summarize = dspy.Predict("text -> summary")

    text = """The Internet of Things (IoT) refers to the network of physical devices, 
    vehicles, home appliances, and other items embedded with electronics, software, 
    sensors, actuators, and connectivity which enables these things to connect and 
    exchange data. This creates opportunities for more direct integration of the 
    physical world into computer-based systems, resulting in efficiency improvements, 
    economic benefits, and reduced human exertions."""

    # Try different models
    models = [("gpt-3.5-turbo", 0.5), ("claude-3-haiku-20240307", 0.3), ("gpt-4", 0.2)]

    for model_name, temp in models:
        # Switch model
        ember_lm = EmberLM(model=model_name, temperature=temp, max_tokens=100)
        dspy.configure(lm=ember_lm)

        # Generate summary
        result = summarize(text=text)
        print(f"Model: {model_name}")
        print(f"Summary: {result.summary}\n")

    # Print usage metrics from last model
    print("Usage Metrics from GPT-4:")
    metrics = ember_lm.get_usage_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def example_custom_signatures():
    """Using custom signatures with type hints."""
    print("=== Custom Signatures Example ===")

    ember_lm = EmberLM(model="claude-3-sonnet-20240229")
    dspy.configure(lm=ember_lm)

    # Define custom signature with descriptions
    class ProductReview(dspy.Signature):
        """Analyze a product review to extract key information."""

        review: str = dspy.InputField(desc="The product review text")
        sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")
        rating: float = dspy.OutputField(desc="Predicted rating from 1.0 to 5.0")
        key_points: list = dspy.OutputField(desc="List of main points mentioned")

    analyze = dspy.Predict(ProductReview)

    review = """I bought this laptop last month and I'm mostly satisfied. 
    The performance is excellent for programming and the battery life is 
    impressive (easily 10+ hours). However, the keyboard feels a bit mushy 
    and the trackpad could be more responsive. Overall, good value for money."""

    result = analyze(review=review)
    print(f"Review: {review}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Rating: {result.rating}")
    print(f"Key Points: {result.key_points}")


if __name__ == "__main__":
    # Run all examples
    example_basic_prediction()
    print("\n" + "=" * 50 + "\n")

    example_chain_of_thought()
    print("\n" + "=" * 50 + "\n")

    example_multi_hop_qa()
    print("\n" + "=" * 50 + "\n")

    example_model_switching()
    print("\n" + "=" * 50 + "\n")

    example_custom_signatures()
