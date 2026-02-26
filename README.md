# Local RAG System (LangChain · Ollama · Postgres · FastAPI)

A fully local Retrieval-Augmented Generation (RAG) service that ingests documents, generates embeddings, stores vectors in Postgres (pgvector), retrieves semantically relevant context, and produces grounded answers using a local LLM.

The system includes:
- CLI ingestion and query tools
- FastAPI service with OpenAPI documentation
- Vector persistence via Postgres + pgvector
- Local LLM execution through Ollama
- Lightweight evaluation with JSON artifact output

The entire pipeline runs locally without external hosted LLM dependencies.

---

## Overview

This project implements a complete RAG workflow:

1. Document ingestion  
2. Text chunking  
3. Embedding generation  
4. Vector storage (pgvector)  
5. Similarity retrieval (Top-K)  
6. Context-augmented prompt construction  
7. Grounded LLM response with citations  
8. Evaluation reporting  

The focus is on production-oriented LLM application design rather than notebook experimentation.

---

## Architecture

Documents  
→ Chunking  
→ Embeddings (Ollama)  
→ Postgres + pgvector  
→ Top-K Retrieval  
→ Prompt Construction  
→ Local LLM (Ollama)  
→ Answer + Citations  

---

## Core Components

- **LLM:** Ollama local model (e.g., `llama3.2:3b`)
- **Embeddings:** Ollama embedding model (e.g., `nomic-embed-text`)
- **Vector Store:** Postgres with pgvector extension
- **Orchestration:** LangChain
- **API Layer:** FastAPI (`/docs`)
- **Evaluation:** Python runner generating structured JSON report

---

## Repository Structure
data/docs/              Example documents for ingestion
src/llm.py              LLM factory configuration
src/pg_rag_cli.py       CLI (ingest / ask)
src/api.py              FastAPI service
src/pg_rag_eval.py      Evaluation runner
outputs/                Evaluation artifacts

---

## Running the System Locally
Prerequisites

Python virtual environment (.venv)

Ollama installed and running locally

Postgres running locally

pgvector extension enabled

.env file with API key

---

## In a separate terminal:

ollama serve

---

## Setup Postgres + pgvector (one-time)
createdb ragdb
psql -h localhost -p 5432 ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"

---

## Create .env in project root:

RAG_API_KEY=change-me-123

---

## Ingest Documents
python -m src.pg_rag_cli ingest --recreate

---

## Run Evaluation
python -m src.pg_rag_eval

---

## Run the API
uvicorn src.api:app --reload --port 8000

---

## Open:

http://127.0.0.1:8000/docs

---

## Example API Requests

Health:

curl -i http://127.0.0.1:8000/health

---

## Ask:

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me-123" \
  -d '{"question":"What databases are mentioned in the docs?","k":4}'

---

## ngest:

curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me-123" \
  -d '{"recreate": true, "chunk_size": 250, "chunk_overlap": 40}'

  ---

  Design Considerations

pgvector over FAISS
Enables durable storage and aligns with enterprise database workflows.

Local LLM execution
Eliminates external API dependency and improves data privacy.

Citations returned with answers
Reduces hallucination and supports grounded responses.

Evaluation runner included
Encourages measurable iteration instead of blind prompt tuning.

Potential Extensions

Retrieval reranking

Streaming LLM responses

Structured JSON outputs

Additional regression tests

UI layer (e.g., Streamlit)

Containerization (Docker)

License

MIT (or preferred license)

