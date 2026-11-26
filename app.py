from __future__ import annotations

import asyncio
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import Runner

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from ingestion.embeddings import EmbeddingModel
from vectorstore.faiss_store import FAISSVectorStore
from pdf_agents.rag_agent import build_rag_agent
from pdf_agents.summarization_agent import build_summarization_agent
from pdf_agents.planner import build_planner
from utils.groq_config import (
    create_groq_async_client,
    create_groq_run_config,
    groq_api_key_is_configured,
)

st.set_page_config(
    page_title="PDF Multi-Agent Analyst",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] { background-color: #161b27; }

    .user-bubble {
        background: #1e3a5f;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 16px;
        margin: 6px 0;
        color: #e8f4fd;
        max-width: 85%;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #1a2332;
        border: 1px solid #2d4a6b;
        border-radius: 12px 12px 12px 2px;
        padding: 10px 16px;
        margin: 6px 0;
        color: #d4e8f7;
        max-width: 90%;
    }
    .source-tag {
        font-size: 0.75rem;
        color: #5a9fd4;
        margin-top: 4px;
    }
    h1 { color: #58a6ff !important; }
    .subtitle { color: #8b9eb0; font-size: 0.95rem; margin-top: -12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _init_session() -> None:
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "embedding_model": None,
        "planner": None,
        "processed_docs": [],
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

def _check_api_key() -> bool:
    return groq_api_key_is_configured()


def _build_pipeline(uploaded_files) -> None:
    """Process uploaded PDFs and wire up all agents."""
    st.session_state.processing = True

    vector_store = FAISSVectorStore(embedding_dim=384)
    embedding_model = EmbeddingModel()

    all_chunks: List[Dict] = []
    errors: List[str] = []
    doc_names: List[str] = []

    progress_bar = st.progress(0, text="Starting document processing…")
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        doc_name = uploaded_file.name
        progress_bar.progress(
            int((i / total) * 60),
            text=f"Extracting text from {doc_name}…",
        )
        try:
            file_bytes = uploaded_file.read()
            pages = load_pdf(file_bytes, doc_name)
            chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=200)
            all_chunks.extend(chunks)
            doc_names.append(doc_name)
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        for err in errors:
            st.error(f" {err}")

    if not all_chunks:
        st.error("No text could be extracted from any uploaded file.")
        st.session_state.processing = False
        progress_bar.empty()
        return

    progress_bar.progress(65, text="Generating embeddings…")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedding_model.embed(texts)

    progress_bar.progress(80, text="Building FAISS index…")
    vector_store.add_chunks(all_chunks, embeddings)

    progress_bar.progress(90, text="Assembling agents…")
    rag_agent = build_rag_agent(vector_store, embedding_model)
    summarization_agent = build_summarization_agent(vector_store)
    planner = build_planner(rag_agent, summarization_agent)

    st.session_state.vector_store = vector_store
    st.session_state.embedding_model = embedding_model
    st.session_state.planner = planner
    st.session_state.processed_docs = doc_names

    progress_bar.progress(100, text="Done!")
    progress_bar.empty()
    st.session_state.processing = False
    st.success(
        f" Processed {len(doc_names)} document(s) — "
        f"{len(all_chunks)} chunks indexed."
    )


def _run_agent(user_query: str) -> str:
    """Run the planner synchronously and return its response text."""
    planner = st.session_state.planner
    if planner is None:
        return "No documents have been processed yet. Please upload PDFs first."

    try:
        async def run_agent() -> str:
            async with create_groq_async_client() as client:
                result = await Runner.run(
                    planner,
                    input=user_query,
                    run_config=create_groq_run_config(client),
                )
                return result.final_output or "The agent did not produce a response."

        return asyncio.run(run_agent())
    except Exception as exc:
        return f"Agent error: {exc}"


with st.sidebar:
    st.markdown("## PDF Multi-Agent Analyst")
    st.markdown(
        '<div class="subtitle">Powered by OpenAI Agents SDK + Groq + FAISS</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    if _check_api_key():
        st.success("Groq API key loaded")
    else:
        st.error(
            "GROQ_API_KEY not found.\n\n"
            "Create a `.env` file with:\n```\nGROQ_API_KEY=your_groq_api_key_here\n```"
        )

    st.divider()

    st.markdown("### Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

    process_btn = st.button(
        "Process Documents",
        disabled=not uploaded_files or not _check_api_key(),
        width="stretch",
    )

    if process_btn and uploaded_files:
        _build_pipeline(uploaded_files)

    if st.session_state.processed_docs:
        st.divider()
        st.markdown("### Indexed Documents")
        for doc in st.session_state.processed_docs:
            st.markdown(f"- `{doc}`")

    st.divider()
    if st.button("Clear Chat History", width="stretch"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown(
        "<small style='color:#506070'>The system automatically routes your "
        "question to the RAG Agent or Summarization Agent based on your intent. "
        "No mode selection needed.</small>",
        unsafe_allow_html=True,
    )

st.markdown("# PDF Multi-Agent Analyst")
st.markdown(
    '<div class="subtitle">Ask questions about your documents or request a summary — '
    'the system routes automatically.</div>',
    unsafe_allow_html=True,
)
st.divider()

chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        if st.session_state.planner is None:
            st.info(
                "Upload PDF documents in the sidebar and click **Process Documents** "
                "to get started."
            )
        else:
            st.info(
                "Documents are ready! Ask a question or request a summary below."
            )
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">User: {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">Assistant: {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

st.divider()
user_input = st.chat_input(
    placeholder="Ask a question about your documents, or type 'summarise the document'…",
    disabled=st.session_state.planner is None,
)

if user_input:
    if not user_input.strip():
        st.warning("Please enter a non-empty message.")
    elif not _check_api_key():
        st.error("Cannot process request: GROQ_API_KEY is missing.")
    else:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input.strip()}
        )

        with st.spinner("Thinking…"):
            response = _run_agent(user_input.strip())

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

        st.rerun()
