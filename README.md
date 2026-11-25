# 📄 PDF Multi-Agent Reasoning System

An intelligent, agent-driven application that enables semantic interaction with PDF documents through a conversational Streamlit interface. The system automatically routes user queries to the appropriate specialist agent — no mode buttons, no manual selection.

---

## Problem Statement

Users often need to both query specific information from and get high-level summaries of large PDF documents. Building a system that understands intent and delegates automatically — rather than requiring users to toggle between "question" and "summary" modes — dramatically improves usability and mirrors how humans naturally interact with information.

---

## Features

- 📤 Upload multiple PDFs simultaneously
- 🔍 RAG-powered contextual question answering with source citations
- 📝 Abstractive summarization using map-reduce for long documents
- 🤖 Automatic intent detection — the system chooses the right agent
- 💬 Persistent conversational chat interface
- ⚡ Embeddings computed once per upload, reused for all queries
- 🔒 API keys managed via `.env`, never committed

---

## Architecture

```
                    Streamlit UI
                         │
                    User Request
                         │
                         ▼
                  Planner Agent
              (openai-agents SDK)
                    │       │
           ─────────         ─────────
           │                         │
           ▼                         ▼
      RAG Agent              Summarization Agent
           │                         │
     search_documents        summarize_documents
           │                    (map-reduce)
           ▼                         │
    FAISS Similarity           Chunk Summaries
       Search                        │
           │                    Combined Summary
           ▼                         │
     Retrieved Chunks               LLM
           │                         │
          LLM                        ▼
           │                 Structured Summary
           ▼
   Grounded Answer
   (with source citation)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Agent Orchestration | OpenAI Agents SDK using Groq's OpenAI-compatible API |
| LLM | GPT-OSS 120B (Groq) |
| PDF Extraction | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (faiss-cpu) |
| Environment | python-dotenv |

---

## Installation

### Prerequisites
- Python 3.10 or higher (Python 3.12 recommended; avoid Python 3.11.0 because it can fail while importing the OpenAI Agents SDK)
- A Groq API key

### Steps

```bash
# 1. Clone or unzip the project
cd pdf_multi_agent

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your Groq API key
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key (required) |
| `GROQ_MODEL` | Groq model ID (optional; defaults to `openai/gpt-oss-120b`) |

Create a key in the [Groq Console](https://console.groq.com/keys), copy `.env.example` to `.env`, and set:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

---

## How to Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. Open the app in your browser.
2. Use the sidebar to upload one or more PDF files.
3. Click **Process Documents** — this extracts text, generates embeddings, and builds the FAISS index.
4. Type naturally in the chat box. No mode selection needed.

---

## Example Questions for RAG Testing

```
What is the main research question addressed in this paper?
Who are the authors and what institution are they from?
What methodology was used in the study?
What datasets were used for evaluation?
What were the main limitations mentioned?
Explain the system architecture described in the document.
What results did the paper report?
Is there any mention of future work?
```

---

## Example Summarization Prompts

```
Summarise the document.
Give me an overview of this paper.
What is this document about?
TL;DR
Key points from the PDF.
What are the main findings?
Give me a high-level summary.
```

---

## Agent Architecture

### Planner Agent
- Receives the user's raw input
- Uses the LLM to decide between `rag_agent` and `summarization_agent`
- Delegates via `agent.as_tool()` — no hardcoded routing logic

### RAG Agent
- Exposes `search_documents(query)` as a function_tool
- Embeds the query and performs FAISS similarity search
- Returns top-k chunks with source metadata to the LLM
- Instructed to answer only from retrieved context

### Summarization Agent
- Exposes `summarize_documents(focus)` as a function_tool
- Implements map-reduce: each chunk is summarised independently, then summaries are combined in batches
- The agent structures the final output into sections

---

## RAG Pipeline

```
User Query
    ↓
Embed query (sentence-transformers)
    ↓
FAISS inner-product search (cosine similarity)
    ↓
Top-5 chunks retrieved with doc_name + page_number
    ↓
Context formatted and passed to the configured Groq model
    ↓
Grounded answer with source citations
```

---

## Project Structure

```
pdf_multi_agent/
│
├── app.py                      # Streamlit entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── pdf_agents/
│   ├── __init__.py
│   ├── planner.py              # Orchestrator agent
│   ├── rag_agent.py            # RAG specialist
│   └── summarization_agent.py # Summarization specialist
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_loader.py           # PyMuPDF text extraction
│   ├── chunker.py              # Overlapping text chunking
│   └── embeddings.py          # Sentence-transformer wrapper
│
├── vectorstore/
│   ├── __init__.py
│   └── faiss_store.py          # FAISS index + metadata
│
└── utils/
    ├── __init__.py
    ├── llm.py                  # Groq chat-completions wrapper
    ├── groq_config.py          # Groq API and Agents SDK provider configuration
    └── prompts.py              # All system/task prompts
```

---

## Limitations

- Scanned PDFs (image-only) are not supported — text extraction requires selectable text
- Very large documents (100+ pages) may be slow to process due to map-reduce LLM calls
- The system uses a Groq-hosted model for agent responses; set `GROQ_MODEL` to another supported model ID to change it
- No persistent storage — the vector store is rebuilt on each session

---

## Future Improvements

- OCR support for scanned PDFs (e.g., Tesseract)
- Persistent FAISS index saved to disk between sessions
- Multi-turn conversation memory for follow-up questions
- Streaming LLM responses in the Streamlit UI
- Support for additional document types (DOCX, TXT, HTML)
- Reranking retrieved chunks with a cross-encoder model
- Confidence scores displayed alongside RAG answers
