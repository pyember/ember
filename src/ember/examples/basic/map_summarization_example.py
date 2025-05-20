"""Map Summarization Example

This example demonstrates how to use the Summarization operator to summarize multiple large documents,
and then use JudgeSynthesis to synthesize the best overall answer to a question based on those summaries.

To run:
    uv run python src/ember/examples/basic/map_summarization_example.py
"""
from ember.core import non

# Example large documents (could be paragraphs, articles, or reports)
documents = [
    """Climate change is primarily driven by the accumulation of greenhouse gases in the atmosphere, 
    resulting from human activities such as burning fossil fuels, deforestation, and industrial processes. 
    These gases trap heat, leading to global temperature rise, melting ice caps, and more extreme weather events.""",
    """The Earth's climate is changing due to increased concentrations of carbon dioxide and other greenhouse gases. 
    Human activities, especially the combustion of coal, oil, and gas, are the main contributors. 
    The consequences include rising sea levels, shifting weather patterns, and threats to biodiversity.""",
    """Scientific consensus indicates that anthropogenic emissions of greenhouse gases are the dominant cause of 
    recent global warming. The Intergovernmental Panel on Climate Change (IPCC) reports that urgent action is needed 
    to mitigate emissions and adapt to unavoidable impacts.""",
    """Rising global temperatures are linked to human-induced emissions from transportation, agriculture, and energy production. 
    These changes disrupt ecosystems, threaten food security, and increase the frequency of natural disasters.""",
    """Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities. 
    The effects are widespread, impacting health, economies, and the environment. International cooperation is essential 
    to address these challenges.""",
    """The burning of fossil fuels releases carbon dioxide, which accumulates in the atmosphere and enhances the greenhouse effect. 
    This leads to global warming, ocean acidification, and more frequent extreme weather events.""",
]

# Step 1: Summarize each document using the Summarization operator
summarizer = non.Summarization(model_name="openai:gpt-4.1-mini", temperature=0.5)

summaries = summarizer(
    inputs=non.MapEnsembleOperatorInputs(items=documents)
)

# Step 2: Synthesize the best answer to a question using JudgeSynthesis
question = "What causes climate change and what are its main impacts?"

judge = non.JudgeSynthesis(model_name="openai:gpt-4.1", temperature=0.7)

judge_inputs = non.JudgeSynthesisInputs(
    query=question,
    responses=summaries.outputs,
)

judge_result = judge(inputs=judge_inputs)

print("=== Synthesized Answer ===")
print("Reasoning:\n", judge_result.reasoning)
print("\nFinal Answer:\n", judge_result.final_answer)
