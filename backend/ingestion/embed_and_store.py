"""
Embeds code chunks using Gemini's embedding model and stores them in Postgres.

Reads chunks from the AST-based chunker, generates a vector embedding for each
chunk (concurrently, with retry/backoff to respect API rate limits), and inserts
each row into the code_chunks table alongside its metadata.
"""

import asyncio
from email.mime import text
import os

import asyncpg
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from chunker import Chunk, chunk_repo

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
EMBEDDING_DIM = 1536
MAX_CONCURRENT_REQUESTS = 2  # stay comfortably under the free tier's RPM limit

client = genai.Client(api_key=GEMINI_API_KEY)


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60))
async def embed_chunk_text(text: str) -> list[float]:
    """Embed a single chunk's code text, with retry/backoff on transient failures."""
    result = await asyncio.to_thread(
        client.models.embed_content,
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return result.embeddings[0].values


async def embed_all_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Embed all chunks concurrently, capped at MAX_CONCURRENT_REQUESTS in flight."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def bound_embed(text: str) -> list[float]:
        async with semaphore:
            result = await embed_chunk_text(text)
            await asyncio.sleep(1.5)  # small pacing delay to stay under RPM limits
            return result

    tasks = [bound_embed(c.code_text) for c in chunks]
    return await asyncio.gather(*tasks)


async def store_chunks(repo_name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Insert all chunks and their embeddings into the code_chunks table."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        for chunk, embedding in zip(chunks, embeddings):
            await conn.execute(
                """
                INSERT INTO code_chunks
                    (repo, filepath, symbol_name, symbol_type, start_line, end_line, code_text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                repo_name,
                chunk.filepath,
                chunk.symbol_name,
                chunk.symbol_type,
                chunk.start_line,
                chunk.end_line,
                chunk.code_text,
                str(embedding),  # asyncpg + pgvector: pass as a string literal like '[0.1,0.2,...]'
            )
    finally:
        await conn.close()


async def main():
    repo_path = "../corvex_data/requests/src/requests"
    repo_name = "requests"

    print("Chunking repo...")
    chunks = chunk_repo(repo_path)
    print(f"Extracted {len(chunks)} chunks.")

    print("Embedding chunks (this may take a bit)...")
    embeddings = await embed_all_chunks(chunks)
    print(f"Embedded {len(embeddings)} chunks.")

    print("Storing in Postgres...")
    await store_chunks(repo_name, chunks, embeddings)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())