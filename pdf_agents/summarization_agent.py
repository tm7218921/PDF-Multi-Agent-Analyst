from __future__ import annotations

from typing import List, TYPE_CHECKING

from agents import Agent, function_tool

from utils.groq_config import get_groq_model
from utils.llm import call_llm
from utils.prompts import (
    SUMMARIZATION_SYSTEM_PROMPT,
    CHUNK_SUMMARY_PROMPT,
    COMBINE_SUMMARIES_PROMPT,
)

if TYPE_CHECKING:
    from vectorstore.faiss_store import FAISSVectorStore


REDUCE_BATCH_SIZE = 10


def _summarise_chunk(chunk_text: str) -> str:
    """Generate a short summary for a single chunk (map step)."""
    prompt = CHUNK_SUMMARY_PROMPT.format(chunk_text=chunk_text)
    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages, max_tokens=256, temperature=0.3)


def _combine_summaries(summaries: List[str]) -> str:
    """Combine a list of chunk summaries into one (reduce step)."""
    combined = "\n\n".join(f"- {s}" for s in summaries)
    prompt = COMBINE_SUMMARIES_PROMPT.format(summaries=combined)
    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages, max_tokens=512, temperature=0.3)


def _map_reduce(chunks: List[str]) -> str:
    """
    Full map-reduce pipeline:
    - Map:    summarise each chunk independently
    - Reduce: batch-combine summaries until one remains
    """
    
    chunk_summaries = [_summarise_chunk(c) for c in chunks]

    while len(chunk_summaries) > 1:
        batched: List[str] = []
        for i in range(0, len(chunk_summaries), REDUCE_BATCH_SIZE):
            batch = chunk_summaries[i : i + REDUCE_BATCH_SIZE]
            if len(batch) == 1:
                batched.append(batch[0])
            else:
                batched.append(_combine_summaries(batch))
        chunk_summaries = batched

    return chunk_summaries[0]


def build_summarization_agent(vector_store: "FAISSVectorStore") -> Agent:
    """
    Constructs and returns the Summarization Agent.
    """

    @function_tool
    def summarize_documents(focus: str = "") -> str:
        """
        Summarize all uploaded PDF documents using a map-reduce strategy.
        Returns a combined intermediate summary for the agent to structure.

        Args:
            focus: Optional topic or aspect to focus the summary on.
                   Leave empty for a general summary.
        """
        if not vector_store.is_ready:
            return (
                "No documents have been uploaded and processed yet. "
                "Please upload PDF files first."
            )

        all_chunks: List[str] = [m["text"] for m in vector_store.metadata]

        if not all_chunks:
            return "No text chunks available in the vector store."

        try:
            intermediate = _map_reduce(all_chunks)
        except Exception as exc:
            return f"Summarization failed during map-reduce: {exc}"

        if focus:
            intermediate = (
                f"Focus area requested: {focus}\n\n"
                f"Document content summary:\n{intermediate}"
            )

        return intermediate

    summarization_agent = Agent(
        name="Summarization Agent",
        instructions=SUMMARIZATION_SYSTEM_PROMPT,
        tools=[summarize_documents],
        model=get_groq_model(),
    )

    return summarization_agent
