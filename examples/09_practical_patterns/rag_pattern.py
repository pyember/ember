"""Retrieval-Augmented Generation (RAG) patterns.

This example demonstrates:
- Basic RAG architecture
- Document chunking strategies
- Retrieval with ranking
- Context injection patterns
- Multi-source retrieval

Run with:
    python examples/09_practical_patterns/rag_pattern.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ember.api import op


# =============================================================================
# Part 1: Document and Chunk Representations
# =============================================================================

@dataclass
class Document:
    """A document with content and metadata."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def chunks(self, size: int = 200, overlap: int = 50) -> List["Chunk"]:
        """Split document into overlapping chunks."""
        chunks = []
        text = self.content
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + size
            chunk_text = text[start:end]

            chunks.append(Chunk(
                id=f"{self.id}_chunk_{chunk_id}",
                document_id=self.id,
                content=chunk_text,
                start_offset=start,
                metadata=self.metadata.copy(),
            ))

            start = end - overlap
            chunk_id += 1

        return chunks


@dataclass
class Chunk:
    """A chunk from a document."""

    id: str
    document_id: str
    content: str
    start_offset: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


def demonstrate_chunking() -> None:
    """Show document chunking strategies."""
    print("Part 1: Document Chunking")
    print("-" * 50)

    doc = Document(
        id="doc_001",
        content=(
            "Retrieval-Augmented Generation (RAG) combines retrieval with generation. "
            "First, relevant documents are retrieved based on the query. "
            "Then, the retrieved context is used to augment the generation process. "
            "This allows the model to access external knowledge beyond its training data."
        ),
        metadata={"source": "rag_intro", "author": "example"},
    )

    print(f"Document: {doc.id}")
    print(f"Content length: {len(doc.content)} characters")
    print()

    # Chunk with different sizes
    for size, overlap in [(100, 20), (50, 10)]:
        chunks = doc.chunks(size=size, overlap=overlap)
        print(f"Chunking (size={size}, overlap={overlap}):")
        print(f"  Number of chunks: {len(chunks)}")
        for chunk in chunks[:2]:
            preview = chunk.content[:40] + "..." if len(chunk.content) > 40 else chunk.content
            print(f"    {chunk.id}: '{preview}'")
        if len(chunks) > 2:
            print(f"    ... and {len(chunks) - 2} more")
        print()


# =============================================================================
# Part 2: Simple Retriever
# =============================================================================

@dataclass
class SimpleRetriever:
    """A simple keyword-based retriever for demonstration."""

    chunks: List[Chunk] = field(default_factory=list)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the index."""
        self.chunks.extend(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Tuple[Chunk, float]]:
        """Retrieve chunks by keyword matching."""
        query_terms = set(query.lower().split())
        results = []

        for chunk in self.chunks:
            chunk_terms = set(chunk.content.lower().split())
            # Simple overlap score
            overlap = len(query_terms & chunk_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                results.append((chunk, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def demonstrate_retrieval() -> None:
    """Show basic retrieval."""
    print("Part 2: Simple Retrieval")
    print("-" * 50)

    # Create documents
    docs = [
        Document(
            id="python_basics",
            content="Python is a high-level programming language. It emphasizes code readability.",
        ),
        Document(
            id="ml_intro",
            content="Machine learning is a subset of AI. It enables systems to learn from data.",
        ),
        Document(
            id="rag_guide",
            content="RAG combines retrieval with generation. It augments LLM responses with context.",
        ),
    ]

    # Build retriever
    retriever = SimpleRetriever()
    for doc in docs:
        retriever.add_chunks(doc.chunks(size=100, overlap=20))

    print(f"Indexed {len(retriever.chunks)} chunks from {len(docs)} documents")
    print()

    # Query
    query = "How does RAG augment LLM responses?"
    results = retriever.retrieve(query, top_k=2)

    print(f"Query: '{query}'")
    print(f"Retrieved {len(results)} chunks:")
    for chunk, score in results:
        print(f"  Score: {score:.2f} | {chunk.id}")
        print(f"    '{chunk.content[:60]}...'")
    print()


# =============================================================================
# Part 3: RAG Pipeline
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    top_k: int = 3
    context_template: str = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    separator: str = "\n---\n"


@dataclass
class RAGPipeline:
    """A complete RAG pipeline."""

    retriever: SimpleRetriever
    config: RAGConfig = field(default_factory=RAGConfig)

    def build_prompt(
        self,
        query: str,
        retrieved: List[Tuple[Chunk, float]],
    ) -> str:
        """Build the augmented prompt."""
        context_parts = []
        for chunk, score in retrieved:
            context_parts.append(f"[Source: {chunk.document_id}, Score: {score:.2f}]")
            context_parts.append(chunk.content)

        context = self.config.separator.join(context_parts)
        return self.config.context_template.format(
            context=context,
            question=query,
        )

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the RAG pipeline."""
        # Retrieve
        retrieved = self.retriever.retrieve(query, top_k=self.config.top_k)

        # Build prompt
        prompt = self.build_prompt(query, retrieved)

        # In production, this would call an LLM
        return {
            "query": query,
            "retrieved_count": len(retrieved),
            "sources": [c.document_id for c, _ in retrieved],
            "prompt": prompt,
            "answer": "[Would be generated by LLM]",
        }


def demonstrate_rag_pipeline() -> None:
    """Show complete RAG pipeline."""
    print("Part 3: RAG Pipeline")
    print("-" * 50)

    # Setup
    docs = [
        Document(id="api_docs", content="The API supports JSON requests. Use POST for creation."),
        Document(id="auth_docs", content="Authentication uses JWT tokens. Tokens expire after 1 hour."),
        Document(id="error_docs", content="Error responses include error codes. 400 means bad request."),
    ]

    retriever = SimpleRetriever()
    for doc in docs:
        retriever.add_chunks(doc.chunks(size=80, overlap=15))

    pipeline = RAGPipeline(
        retriever=retriever,
        config=RAGConfig(top_k=2),
    )

    # Execute
    query = "How do I authenticate with the API?"
    result = pipeline.run(query)

    print(f"Query: {result['query']}")
    print(f"Retrieved from: {result['sources']}")
    print(f"\nAugmented prompt:\n{result['prompt'][:300]}...")
    print()


# =============================================================================
# Part 4: Context Window Management
# =============================================================================

@dataclass
class ContextManager:
    """Manage context within token limits."""

    max_tokens: int = 1000
    chars_per_token: float = 4.0  # Rough estimate

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return int(len(text) / self.chars_per_token)

    def fit_context(
        self,
        chunks: List[Tuple[Chunk, float]],
        query: str,
        template_overhead: int = 100,
    ) -> List[Tuple[Chunk, float]]:
        """Select chunks that fit within token limit."""
        available = self.max_tokens - template_overhead - self.estimate_tokens(query)

        fitted = []
        used = 0

        for chunk, score in chunks:
            chunk_tokens = self.estimate_tokens(chunk.content)
            if used + chunk_tokens <= available:
                fitted.append((chunk, score))
                used += chunk_tokens
            else:
                break

        return fitted


def demonstrate_context_management() -> None:
    """Show context window management."""
    print("Part 4: Context Window Management")
    print("-" * 50)

    manager = ContextManager(max_tokens=200)

    # Simulate retrieved chunks
    chunks = [
        (Chunk(id="c1", document_id="d1", content="A" * 400, start_offset=0), 0.9),
        (Chunk(id="c2", document_id="d2", content="B" * 200, start_offset=0), 0.8),
        (Chunk(id="c3", document_id="d3", content="C" * 100, start_offset=0), 0.7),
    ]

    print(f"Token limit: {manager.max_tokens}")
    print(f"Chunks before fitting: {len(chunks)}")

    for chunk, score in chunks:
        tokens = manager.estimate_tokens(chunk.content)
        print(f"  {chunk.id}: ~{tokens} tokens, score={score}")

    fitted = manager.fit_context(chunks, "Test query")
    print(f"\nChunks after fitting: {len(fitted)}")
    for chunk, score in fitted:
        print(f"  {chunk.id}: kept")
    print()


# =============================================================================
# Part 5: Multi-Source Retrieval
# =============================================================================

@dataclass
class MultiSourceRetriever:
    """Retrieve from multiple sources and merge results."""

    sources: Dict[str, SimpleRetriever] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

    def add_source(
        self,
        name: str,
        retriever: SimpleRetriever,
        weight: float = 1.0,
    ) -> None:
        """Add a retrieval source."""
        self.sources[name] = retriever
        self.weights[name] = weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float, str]]:
        """Retrieve from all sources and merge."""
        all_results = []

        for name, retriever in self.sources.items():
            weight = self.weights.get(name, 1.0)
            results = retriever.retrieve(query, top_k=top_k)

            for chunk, score in results:
                weighted_score = score * weight
                all_results.append((chunk, weighted_score, name))

        # Sort by weighted score
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]


def demonstrate_multi_source() -> None:
    """Show multi-source retrieval."""
    print("Part 5: Multi-Source Retrieval")
    print("-" * 50)

    # Create source-specific retrievers
    docs_retriever = SimpleRetriever()
    docs_retriever.add_chunks([
        Chunk(id="doc_1", document_id="docs", content="API documentation for endpoints", start_offset=0),
    ])

    faq_retriever = SimpleRetriever()
    faq_retriever.add_chunks([
        Chunk(id="faq_1", document_id="faq", content="FAQ about API authentication", start_offset=0),
    ])

    # Multi-source retriever with weights
    multi = MultiSourceRetriever()
    multi.add_source("documentation", docs_retriever, weight=1.0)
    multi.add_source("faq", faq_retriever, weight=0.8)  # FAQs weighted lower

    query = "API documentation"
    results = multi.retrieve(query, top_k=3)

    print(f"Query: '{query}'")
    print("Results from multiple sources:")
    for chunk, score, source in results:
        print(f"  [{source}] {chunk.id}: score={score:.2f}")
        print(f"    '{chunk.content}'")
    print()


# =============================================================================
# Part 6: RAG Best Practices
# =============================================================================

def demonstrate_best_practices() -> None:
    """Show RAG best practices."""
    print("Part 6: RAG Best Practices")
    print("-" * 50)

    practices = [
        (
            "Chunk size matters",
            "Smaller chunks = more precise retrieval. Larger = more context.",
        ),
        (
            "Overlap prevents boundary issues",
            "10-20% overlap ensures information isn't split awkwardly.",
        ),
        (
            "Metadata enables filtering",
            "Store source, date, type to filter before/after retrieval.",
        ),
        (
            "Rerank retrieved results",
            "Use a cross-encoder to reorder initial retrieval results.",
        ),
        (
            "Manage context window",
            "Fit highest-scored chunks within token limits.",
        ),
        (
            "Cite sources in output",
            "Include source references so users can verify.",
        ),
    ]

    for i, (title, description) in enumerate(practices, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
    print()


def main() -> None:
    """Demonstrate RAG patterns."""
    print("Retrieval-Augmented Generation Patterns")
    print("=" * 50)
    print()

    demonstrate_chunking()
    demonstrate_retrieval()
    demonstrate_rag_pipeline()
    demonstrate_context_management()
    demonstrate_multi_source()
    demonstrate_best_practices()

    print("Key Takeaways")
    print("-" * 50)
    print("1. RAG combines retrieval with generation for grounded responses")
    print("2. Chunking strategy affects retrieval precision")
    print("3. Context window management prevents token overflow")
    print("4. Multi-source retrieval enables diverse knowledge")
    print("5. Metadata and filtering improve relevance")
    print("6. Always cite sources for verifiability")
    print()
    print("Next: See structured_output.py for output extraction patterns")


if __name__ == "__main__":
    main()
