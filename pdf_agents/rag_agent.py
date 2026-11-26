
from __future__ import annotations

from typing import TYPE_CHECKING

from agents import Agent, function_tool

from utils.groq_config import get_groq_model
from utils.prompts import RAG_SYSTEM_PROMPT

if TYPE_CHECKING:
    from vectorstore.faiss_store import FAISSVectorStore
    from ingestion.embeddings import EmbeddingModel

TOP_K = 5


def build_rag_agent(
    vector_store: "FAISSVectorStore",
    embedding_model: "EmbeddingModel",
) -> Agent:
    """
    Constructs and returns the RAG Agent.

    The search_documents tool is defined inside this function so it closes
    over the shared vector_store and embedding_model instances.
    """

    @function_tool
    def search_documents(query: str) -> str:
        """
        Search the uploaded PDF documents for content relevant to the query.
        Returns the top matching chunks with their source document and page number.

        Args:
            query: The user's question or search terms.
        """
        if not vector_store.is_ready:
            return (
                "No documents have been uploaded and processed yet. "
                "Please upload PDF files first."
            )

        try:
            query_embedding = embedding_model.embed_query(query)
            results = vector_store.search(query_embedding, top_k=TOP_K)
        except Exception as exc:
            return f"Search failed: {exc}"

        if not results:
            return "No relevant content found in the uploaded documents."

        formatted_chunks: list[str] = []
        for i, chunk in enumerate(results, start=1):
            source_info = f"[Source: {chunk['doc_name']}, Page {chunk['page_number']}]"
            formatted_chunks.append(
                f"--- Chunk {i} {source_info} ---\n{chunk['text']}"
            )

        return "\n\n".join(formatted_chunks)

    rag_agent = Agent(
        name="RAG Agent",
        instructions=RAG_SYSTEM_PROMPT,
        tools=[search_documents],
        model=get_groq_model(),
    )

    return rag_agent
