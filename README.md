# Local RAG System (Ollama + LangChain + pgvector + FastAPI)

A local Retrieval-Augmented Generation (RAG) application that:
- ingests local text docs
- chunks + embeds them
- stores embeddings in Postgres (pgvector)
- retrieves relevant context for a user query
- generates a grounded answer using a local LLM (Ollama)
- returns citations (source + snippet)
- includes a lightweight evaluation runner that writes an eval report JSON

This project is designed to demonstrate production-style LLM application engineering:
**RAG pipeline + vector database + CLI + API + evaluation + artifacts**.

---

## What This Proves (Hiring-Manager Friendly)
- You can build an end-to-end RAG pipeline (ingest → embed → store → retrieve → answer)
- You understand grounding + citations (reducing hallucinations)
- You can ship both notebook/prototype AND a CLI + API (not “just a notebook”)
- You can add evaluation + saved artifacts (`outputs/eval_report.json`)

---

## Architecture (High Level)

Docs → Chunking → Embeddings → pgvector (Postgres)
User Question → Retrieval (Top-K) → Prompt with Context → Ollama LLM → Answer + Citations

**Key components**
- **LLM**: Ollama local model (e.g., `llama3.2:3b`)
- **Embeddings**: Ollama embedding model (e.g., `nomic-embed-text`)
- **Vector DB**: Postgres + pgvector
- **Orchestration**: LangChain
- **API**: FastAPI (OpenAPI docs at `/docs`)
- **Eval**: Python runner saving JSON report

---

## Repo Structure

- `data/docs/`  
  Example documents used for ingestion

- `src/llm.py`  
  LLM factory (Ollama base_url set to prevent connection issues)

- `src/pg_rag_cli.py`  
  CLI for ingest + ask

- `src/api.py`  
  FastAPI service: `/health`, `/ingest`, `/ask`

- `src/pg_rag_eval.py`  
  Evaluation runner writing `outputs/eval_report.json`

- `outputs/`  
  Generated artifacts (eval report, etc.)

---

## How to Run (Local)

### Prereqs
- Python venv: `.venv` (already set up)
- Ollama running locally
- Postgres running locally + pgvector enabled
- API key set in `.env`

### 1) Start Ollama
In one terminal tab:
```bash
ollama serve
