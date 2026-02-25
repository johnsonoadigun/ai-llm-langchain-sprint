# src/rag.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Dict

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm

# --- Project paths (robust) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # .../ai-llm-langchain-sprint
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INDEX_DIR = OUTPUTS_DIR / "faiss_index"


PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using the context.
If the answer is explicitly present in the context, extract it directly.
If it is not present, say: "I don't know based on the context."

Question:
{question}

Context:
{context}
"""
)


def load_docs(docs_dir: Path = DOCS_DIR) -> List[Dict[str, str]]:
    """Load all .txt docs from data/docs into a simple list."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    files = sorted(docs_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {docs_dir}")

    docs: List[Dict[str, str]] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({"source": str(f), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Simple chunking (fast + good enough for sprint)."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_or_load_vs(
    index_dir: Path = DEFAULT_INDEX_DIR,
    embed_model: str = "nomic-embed-text",
) -> FAISS:
    """Load FAISS index if it exists; otherwise build it from docs and save."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    embeddings = OllamaEmbeddings(model=embed_model)

    if index_dir.exists() and any(index_dir.iterdir()):
        # Load existing persisted index
        vs = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vs

    # Build new index
    docs = load_docs(DOCS_DIR)

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for d in docs:
        source = d["source"]
        for i, chunk in enumerate(chunk_text(d["text"])):
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    # Persist it
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    return vs


def make_context_and_citations(retrieved_docs, max_snip: int = 240):
    """Build context text + lightweight citations."""
    citations = []
    lines = []

    for d in retrieved_docs:
        src = d.metadata.get("source", "unknown")
        txt = (d.page_content or "").strip()
        snip = txt[:max_snip].replace("\n", " ")

        lines.append(f"- {txt}")
        citations.append({"source": src, "snippet": snip})

    context = "\n\n".join(lines)
    return context, citations


def rag_answer(
    question: str,
    k: int = 4,
    index_dir: Path = DEFAULT_INDEX_DIR,
    llm_model: str = "llama3.2:3b",
    embed_model: str = "nomic-embed-text",
) -> Dict[str, Any]:
    vs = build_or_load_vs(index_dir=index_dir, embed_model=embed_model)

    retriever = vs.as_retriever(search_kwargs={"k": k})

    # New API: retriever.invoke()
    retrieved = retriever.invoke(question)

    context, citations = make_context_and_citations(retrieved)

    llm = get_llm(model=llm_model, temperature=0.0)

    # Robust: always extract content + fallback
    result = (PROMPT | llm).invoke({"question": question, "context": context})
    answer = getattr(result, "content", str(result)).strip()
    if not answer:
        answer = "I don't know based on the context."

    return {
        "question": question,
        "answer": answer,
        "k": k,
        "index_dir": str(index_dir),
        "llm_model": llm_model,
        "embed_model": embed_model,
        "citations": citations,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple RAG CLI over local docs with FAISS + Ollama.")
    parser.add_argument("question", type=str, help="Question to ask the RAG system.")
    parser.add_argument("--k", type=int, default=4, help="Top-k chunks to retrieve.")
    parser.add_argument("--index_dir", type=str, default=str(DEFAULT_INDEX_DIR), help="Where FAISS index is stored.")
    parser.add_argument("--llm_model", type=str, default="llama3.2:3b", help="Ollama chat model name.")
    parser.add_argument("--embed_model", type=str, default="nomic-embed-text", help="Ollama embedding model name.")
    args = parser.parse_args()

    result = rag_answer(
        args.question,
        k=args.k,
        index_dir=Path(args.index_dir),
        llm_model=args.llm_model,
        embed_model=args.embed_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()