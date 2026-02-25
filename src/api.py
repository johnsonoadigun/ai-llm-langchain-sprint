from __future__ import annotations

import os
import time
import uuid
import logging
from collections import OrderedDict
from typing import Any, Tuple

from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from src.pg_rag_cli import ingest as ingest_docs
from src.pg_rag_cli import ask as ask_rag

load_dotenv()

app = FastAPI(title="Local RAG API (pgvector + Ollama)")

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.time()

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = round((time.time() - start) * 1000, 2)
            logger.exception(
                f"request_id={request_id} method={request.method} path={request.url.path} status=500 latency_ms={elapsed_ms}"
            )
            raise

        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.info(
            f"request_id={request_id} method={request.method} path={request.url.path} "
            f"status={response.status_code} latency_ms={elapsed_ms}"
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-ms"] = str(elapsed_ms)
        return response


app.add_middleware(RequestIdMiddleware)

# ---------------- Auth ----------------
API_KEY = os.getenv("RAG_API_KEY", "").strip()
if not API_KEY:
    logger.warning("RAG_API_KEY is not set. Set it in .env for API key protection.")


def require_api_key(x_api_key: str | None):
    if not API_KEY:
        # If user hasn't set a key, don't hard-block; but warn in logs.
        logger.warning("API key not enforced because RAG_API_KEY is empty.")
        return
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------- Bounded TTL Cache (for /ask) ----------------
class TTLCache:
    """
    Bounded in-memory cache:
    - maxsize limits memory
    - ttl_seconds expires entries
    """

    def __init__(self, maxsize: int = 200, ttl_seconds: int = 600):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[Tuple[str, int], Tuple[float, Any]] = OrderedDict()

    def get(self, key: Tuple[str, int]):
        now = time.time()
        if key not in self._store:
            return None

        ts, val = self._store[key]
        if now - ts > self.ttl_seconds:
            # expired
            del self._store[key]
            return None

        # mark as recently used
        self._store.move_to_end(key)
        return val

    def set(self, key: Tuple[str, int], value: Any):
        now = time.time()
        self._store[key] = (now, value)
        self._store.move_to_end(key)

        # evict LRU if over maxsize
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def stats(self):
        return {"size": len(self._store), "maxsize": self.maxsize, "ttl_seconds": self.ttl_seconds}


ask_cache = TTLCache(maxsize=200, ttl_seconds=600)


# ---------------- Schemas ----------------
class IngestRequest(BaseModel):
    recreate: bool = False
    chunk_size: int = 250
    chunk_overlap: int = 40


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=20)


# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"status": "ok", "cache": ask_cache.stats(), "auth_enabled": bool(API_KEY)}


@app.post("/ingest")
def ingest(req: IngestRequest, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    result = ingest_docs(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap, recreate=req.recreate)
    logger.info(
        f"ingest recreate={req.recreate} chunk_size={req.chunk_size} chunk_overlap={req.chunk_overlap} num_chunks={result.get('num_chunks')}"
    )
    # Ingest changes answers; safest is to clear cache by resetting it
    global ask_cache
    ask_cache = TTLCache(maxsize=ask_cache.maxsize, ttl_seconds=ask_cache.ttl_seconds)
    return result


@app.post("/ask")
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)

    key = (req.question.strip(), int(req.k))
    cached = ask_cache.get(key)
    if cached is not None:
        logger.info(f"ask cache_hit=true question_len={len(req.question)} k={req.k}")
        return {"cached": True, **cached}

    result = ask_rag(req.question, k=req.k)
    sources = [c.get("source") for c in result.get("citations", [])]
    logger.info(f"ask cache_hit=false question_len={len(req.question)} k={req.k} sources={sources}")

    ask_cache.set(key, result)
    return {"cached": False, **result}