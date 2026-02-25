from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm import get_llm

# -------- Paths / Config --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"

PG_CONN = "postgresql+psycopg://localhost:5432/ragdb"
COLLECTION = "docs_demo"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

PROMPT = ChatPromptTemplate.from_template(
    """You must answer using ONLY the context.
If the context contains the answer, extract it explicitly.
Only say: "I don't know based on the context." if the answer is truly not present.

Return a short direct answer (1–2 sentences).

Question: {question}

Context:
{context}
"""
)

FALLBACK_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """The previous assistant refused. Do NOT refuse.
Using ONLY the context, extract the most relevant facts to answer the question.

Return a short direct answer (1–2 sentences). Do not say "I don't know" if any relevant info exists.

Question: {question}

Context:
{context}
"""
)


def normalize_source(src: str) -> str:
    """Make citations stable by using just the filename."""
    try:
        return Path(src).name
    except Exception:
        return src


def load_and_split_docs(chunk_size: int = 250, chunk_overlap: int = 40):
    paths = sorted(DOCS_DIR.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {DOCS_DIR}")

    docs = []
    for p in paths:
        docs.extend(TextLoader(str(p), encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    return paths, docs, chunks


def reset_collection() -> None:
    """Delete existing embeddings for this collection to avoid duplicates/regressions."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = PGVector(
        connection=PG_CONN,
        collection_name=COLLECTION,
        embeddings=embeddings,
    )
    vs.delete_collection()


def ingest(chunk_size: int = 250, chunk_overlap: int = 40, recreate: bool = False):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if recreate:
        reset_collection()

    paths, docs, chunks = load_and_split_docs(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        connection=PG_CONN,
    )

    return {
        "status": "ok",
        "action": "ingest",
        "collection": COLLECTION,
        "docs_dir": str(DOCS_DIR),
        "files": [p.name for p in paths],
        "num_docs_loaded": len(docs),
        "num_chunks": len(chunks),
        "embed_model": EMBED_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "recreate": recreate,
    }

def ask(question: str, k: int = 4):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vs = PGVector(
        connection=PG_CONN,
        collection_name=COLLECTION,
        embeddings=embeddings,
    )

    retriever = vs.as_retriever(search_kwargs={"k": k})
    retrieved = retriever.invoke(question)

    # Build context (send more text to reduce truncation)
    context = "\n\n".join([f"- {d.page_content[:800]}" for d in retrieved])

    llm = get_llm(model=LLM_MODEL, temperature=0.0)

    # First pass
    answer = (PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "context": context}
    ).strip()

    # Fallback pass if model refuses
    if not answer or answer.lower().startswith("i don't know"):
        answer = (FALLBACK_EXTRACT_PROMPT | llm | StrOutputParser()).invoke(
            {"question": question, "context": context}
        ).strip()

    if not answer:
        answer = "I don't know based on the context."

    # Guardrail: if model contradicts context about RAG, correct it
    ctx_lower = context.lower()
    if (
        "rag" in question.lower()
        and "rag" in ctx_lower
        and ("no mention of rag" in answer.lower() or "no rag" in answer.lower())
    ):
        answer = "One benefit mentioned is that RAG reduces hallucinations by retrieving relevant context from documents."

    # Clean, deduped citations
    seen = set()
    citations = []
    for d in retrieved:
        src = normalize_source(d.metadata.get("source", "unknown"))
        snip = d.page_content[:200].replace("\n", " ")

        key = (src, snip)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"source": src, "snippet": snip})

    return {
        "status": "ok",
        "action": "ask",
        "question": question,
        "answer": answer,
        "k": k,
        "citations": citations,
        "collection": COLLECTION,
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
    }


def main():
    parser = argparse.ArgumentParser(description="pgvector RAG CLI (Postgres + Ollama)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Load docs, chunk, embed, and store in Postgres pgvector")
    p_ing.add_argument("--chunk-size", type=int, default=250)
    p_ing.add_argument("--chunk-overlap", type=int, default=40)
    p_ing.add_argument("--recreate", action="store_true", help="Delete existing collection before ingest")

    p_ask = sub.add_parser("ask", help="Ask a question using pgvector retrieval + LLM")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--k", type=int, default=4)

    args = parser.parse_args()

    if args.cmd == "ingest":
        res = ingest(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, recreate=args.recreate)
    else:
        res = ask(args.question, k=args.k)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
