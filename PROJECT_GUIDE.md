# PDF Multi-Agent Analyst: Project Guide

This guide describes the current implementation: what each project file is for,
how a request moves through the system, and what the app does and does not do.

## What the system does

The app lets a user upload one or more text-based PDFs, process them into a
searchable in-memory index, and ask questions or request summaries in a chat
interface. The user does not choose an agent. An OpenAI Agents SDK planner
decides whether to call the RAG agent or the summarization agent. Groq provides
the language model through its OpenAI-compatible API.

The app is retrieval-augmented: it retrieves document text before answering
document questions. It is not a model trained or fine-tuned on the uploaded
PDFs.

## Request and document flow

### When PDFs are processed

1. The Streamlit file uploader accepts multiple PDF files.
2. `ingestion/pdf_loader.py` extracts selectable text page by page. Empty pages
   are skipped; PDFs without extractable text are rejected.
3. `ingestion/chunker.py` splits each page into overlapping text chunks.
4. `ingestion/embeddings.py` converts each chunk to a normalized embedding
   using `all-MiniLM-L6-v2`.
5. `vectorstore/faiss_store.py` stores those vectors in a FAISS index and
   keeps each chunk's document name and page number as metadata.
6. `app.py` creates the RAG agent, summarization agent, and planner using the
   vector store, then keeps them in Streamlit session state.

### When the user sends a message

1. The chat UI records the message and shows a progress spinner while the
   request runs.
2. `app.py` sends the current message to the planner through the OpenAI Agents
   SDK. The run uses an asynchronous Groq client that is created and closed for
   that run.
3. The planner prompt directs it to call the RAG agent for a specific document
   question or the summarization agent for a summary/overview. If intent is
   ambiguous, the prompt says to prefer RAG.
4. For a RAG request, `pdf_agents/rag_agent.py` embeds the query, retrieves up
   to five chunks from FAISS, and gives their text and source metadata to the
   agent. The prompt tells it to answer from that context and cite document
   names and page numbers when possible.
5. For a summary request, `pdf_agents/summarization_agent.py` takes all indexed
   chunks, summarizes each chunk, and repeatedly combines batches of summaries
   until one remains. The agent formats the result as a high-level summary.
6. The response is added to the visible chat history.

## Files and folders

### Application and configuration

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point. Sets up the interface and session state, processes uploaded PDFs, creates the agents, runs the planner, and displays chat responses. |
| `requirements.txt` | Python package dependencies installed for local use or deployment. |
| `.env.example` | Safe template showing the `GROQ_API_KEY` and optional `GROQ_MODEL` variable names. Copy it to `.env` for local development and replace the placeholder locally. |
| `.env` | Local secrets file, ignored by Git. It should contain a private API key only on your machine; never commit or share it. |
| `.gitignore` | Excludes `.env`, virtual environments, caches, uploaded PDFs, and local generated index files from Git. |
| `.vscode/settings.json` | Local VS Code Python interpreter and terminal environment settings. This folder is ignored by the current `.gitignore`, so this file is not part of the tracked project. |
| `README.md` | Project overview, setup instructions, technology choices, and architecture summary. |
| `PROJECT_GUIDE.md` | This file: a detailed explanation of project files and runtime behavior. |

### Agent implementation

| File | Purpose |
|---|---|
| `pdf_agents/__init__.py` | Exposes the agent builder functions as package imports. |
| `pdf_agents/planner.py` | Builds the planner and exposes the RAG and summarization agents to it as AgentSDK tools using `agent.as_tool()`. |
| `pdf_agents/rag_agent.py` | Builds the document-question-answering agent and its `search_documents` tool, which embeds the query and searches FAISS. |
| `pdf_agents/summarization_agent.py` | Builds the summarization agent and its `summarize_documents` tool. Implements the map-reduce summary pipeline. |

### PDF ingestion and retrieval

| File | Purpose |
|---|---|
| `ingestion/__init__.py` | Exposes the PDF loader, text chunker, and embedding wrapper. |
| `ingestion/pdf_loader.py` | Uses PyMuPDF to extract text and page numbers from uploaded PDFs. It does not perform OCR. |
| `ingestion/chunker.py` | Splits page text into overlapping chunks and retains document and page metadata. |
| `ingestion/embeddings.py` | Loads `all-MiniLM-L6-v2` and creates normalized embeddings for chunks and user queries. |
| `vectorstore/__init__.py` | Exposes the FAISS vector-store class. |
| `vectorstore/faiss_store.py` | Stores normalized vectors in a FAISS inner-product index and returns the most similar chunks with metadata. |

### Prompts and model connection

| File | Purpose |
|---|---|
| `utils/__init__.py` | Exposes shared LLM and prompt helpers. |
| `utils/prompts.py` | Contains the planner, RAG, summarization, and map-reduce prompt text that guides agent behavior and output. |
| `utils/groq_config.py` | Reads Groq key/model configuration, creates the per-run asynchronous Groq client, and configures the Agents SDK to use the OpenAI-compatible Chat Completions API. |
| `utils/llm.py` | Makes synchronous Groq chat-completion calls used by the map-reduce summarizer. |

## What it does not currently do

- **It does not remember earlier turns when generating a new answer.** Chat
  messages are displayed from Streamlit session state, but `app.py` sends only
  the latest user message to the planner. Follow-up prompts should repeat
  necessary context.
- **It does not read scanned/image-only PDFs.** Those require OCR, which is not
  implemented.
- **It does not persist uploaded documents or the FAISS index.** They are held
  in memory for the active Streamlit session and need to be uploaded and
  processed again in a new session or after a restart.
- **A requested summary focus is not applied during chunk summarization.** The
  focus text is added after the map-reduce summary is generated, so the result
  may not deeply prioritize that topic.
- **It does not guarantee factual correctness.** The RAG prompt instructs the
  agent to stay grounded in retrieved text, but model output should still be
  checked against the cited PDF pages.
- **It does not provide an authentication or per-user quota layer.** If the
  deployed app is publicly accessible, visitors may use the configured Groq
  key and its quota.

## Run locally on Windows

From the project directory, create and activate a virtual environment, install
the dependencies, create `.env` from `.env.example`, and set your own Groq key:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally with your key, then start the app:

```powershell
streamlit run app.py
```

For Streamlit Community Cloud, put `GROQ_API_KEY` and (optionally)
`GROQ_MODEL` in the app's Secrets settings instead of committing `.env`.
