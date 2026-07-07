"""
Retrieval functions for Corvex: embeds a natural-language query and searches
the code_chunks table. Vector-only search first; keyword and hybrid merge
come next.
"""

import asyncio
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EMBEDDING_DIM = 1536

client = genai.Client(api_key=GEMINI_API_KEY)


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60))
async def embed_query(query: str) -> list[float]:
    """Embed a user's natural-language question for retrieval.

    Uses task_type="RETRIEVAL_QUERY" — distinct from the "RETRIEVAL_DOCUMENT"
    type used during ingestion, per Gemini's asymmetric embedding guidance.
    """
    result = await asyncio.to_thread(
        client.models.embed_content,
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return result.embeddings[0].values


async def vector_search(db: AsyncSession, query: str, k: int = 10) -> list[dict]:
    """Embed the query and return the top-k chunks by cosine distance."""
    query_embedding = await embed_query(query)
    embedding_str = str(query_embedding)  # e.g. "[0.1, 0.2, ...]" — asyncpg/pgvector accepts this as a vector literal

    result = await db.execute(
        text("""
            SELECT
                chunk_id, repo, filepath, symbol_name, symbol_type,
                start_line, end_line, code_text,
                embedding <=> CAST(:query_embedding AS vector) AS distance
            FROM code_chunks
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :k
        """),
        {"query_embedding": embedding_str, "k": k},
    )
    rows = result.mappings().all()
    return [dict(row) for row in rows]


async def keyword_search(db: AsyncSession, query: str, k: int = 10) -> list[dict]:
    """Full-text search over code_chunks using the generated tsv column.

    Uses websearch_to_tsquery rather than plainto_tsquery or raw to_tsquery:
    it safely handles arbitrary user input (no syntax errors on malformed
    queries, unlike to_tsquery) while still supporting quoted phrases and
    OR/exclusion operators that plainto_tsquery can't express — the right
    tool for a public-facing search endpoint over natural-language input.
    """
    result = await db.execute(
        text("""
            SELECT
                chunk_id, repo, filepath, symbol_name, symbol_type,
                start_line, end_line, code_text,
                ts_rank(tsv, websearch_to_tsquery('english', :query)) AS rank
            FROM code_chunks
            WHERE tsv @@ websearch_to_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :k
        """),
        {"query": query, "k": k},
    )
    rows = result.mappings().all()
    return [dict(row) for row in rows]


async def hybrid_search(db: AsyncSession, query: str, k: int = 10, rrf_k: int = 60) -> list[dict]:
    """Merge vector and keyword search results via Reciprocal Rank Fusion.

    RRF fuses by rank position, not raw score — vector cosine distance and
    ts_rank live on entirely different, non-comparable scales, so combining
    raw scores directly would be meaningless. Fusing by position sidesteps
    that problem entirely.

    Either leg may legitimately return fewer than k results (or zero, e.g.
    keyword_search on a query with no matching terms) — this function must
    not assume both lists are always full.
    """
    # Fetch a slightly larger candidate pool per leg than the final k, so
    # fusion has enough material to work with before truncating to top-k.
    candidate_k = max(k * 2, 20)

    vector_results = await vector_search(db, query, k=candidate_k)
    keyword_results = await keyword_search(db, query, k=candidate_k)

    scores: dict[int, float] = {}
    chunks_by_id: dict[int, dict] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunks_by_id[cid] = chunk

    for rank, chunk in enumerate(keyword_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunks_by_id[cid] = chunk  # keyword result overwrites vector's copy of same chunk — fine, same row

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    return [
        {**chunks_by_id[cid], "rrf_score": scores[cid]}
        for cid in ranked_ids[:k]
    ]