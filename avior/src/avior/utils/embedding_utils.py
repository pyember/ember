from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Protocol
import math


###########################################################
# 1) Embedding Model Interfaces & Implementations
###########################################################

class EmbeddingModel(Protocol):
    """
    A minimal interface for an embedding model that takes text and returns
    a list of floats as the embedding vector.

    By using a protocol (or ABC), we avoid assumptions about the actual model:
    - Could be a local LLM, an external API call, or a custom neural net.
    """

    def embed_text(self, text: str) -> List[float]:
        """
        Compute an embedding vector for the given text.
        Must return a list of floats (the embedding).
        """
        ...


class MockEmbeddingModel:
    """
    A simple, mock embedding model that does a naive character-level encoding.
    Only for demonstration or testing. Replace with a real model.

    Adheres to the EmbeddingModel interface.
    """
    def embed_text(self, text: str) -> List[float]:
        """
        Convert each character to an ASCII code normalized by some constant.
        """
        if not text:
            return []
        # Example: sum ASCII codes, or store them as floats.
        return [ord(ch) / 256.0 for ch in text]


###########################################################
# 2) Similarity Metric Interface & Implementations
###########################################################

class SimilarityMetric(ABC):
    """
    Base class for similarity metrics between two embedding vectors.
    Derived classes must implement similarity(vector1, vector2) -> float
    """

    @abstractmethod
    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute a similarity score between two embedding vectors.
        Usually returns a value in [0,1] or [-1, 1], depending on metric.
        """
        ...


class CosineSimilarity(SimilarityMetric):
    """
    Standard cosine similarity: 
       sim(a,b) = (a · b) / (||a|| * ||b||)
    """

    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b:
            # No data => similarity is 0 or undefined. We pick 0 for convenience.
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


###########################################################
# 3) High-Level Utility Function
###########################################################

def calculate_text_similarity(
    text1: str,
    text2: str,
    model: EmbeddingModel,
    metric: SimilarityMetric
) -> float:
    """
    Given two pieces of text, an EmbeddingModel, and a SimilarityMetric,
    compute the similarity score. This function demonstrates
    how to keep logic simple & composable.

    Returns a floating-point similarity measure, e.g. [0..1] for CosineSimilarity.
    """
    embedding1 = model.embed_text(text1)
    embedding2 = model.embed_text(text2)
    return metric.similarity(embedding1, embedding2)


###########################################################
# 4) Example Usage (If Running This File Directly)
###########################################################
if __name__ == "__main__":
    mock_model = MockEmbeddingModel()
    cosine = CosineSimilarity()

    text_a = "Hello world!"
    text_b = "Hello, world??"

    score = calculate_text_similarity(text_a, text_b, mock_model, cosine)
    print(f"Similarity between '{text_a}' and '{text_b}': {score}")