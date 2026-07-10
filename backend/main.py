from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import logging
from dotenv import load_dotenv
import os

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

from pydantic import BaseModel
from retrieval import hybrid_search

from fastapi.responses import StreamingResponse
import json

from generation import generate_answer, extract_citations

from fastapi.middleware.cors import CORSMiddleware

import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

RAW_DATABASE_URL = os.environ["DATABASE_URL"]

# Neon gives a postgresql:// URL; async SQLAlchemy needs the asyncpg driver in the scheme.
ASYNC_DATABASE_URL = RAW_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# asyncpg doesn't accept the libpq-style "sslmode" query param (Neon includes this
# by default) — strip it out since we set SSL explicitly via connect_args instead.
def _strip_sslmode(url: str) -> str:
    parts = urlsplit(url)
    query_pairs = [(k, v) for k, v in parse_qsl(parts.query) if k != "sslmode"]
    new_query = urlencode(query_pairs)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))

ASYNC_DATABASE_URL = _strip_sslmode(ASYNC_DATABASE_URL)

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    connect_args={"ssl": "require"},
    pool_pre_ping=True,
)

async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_db():
    async with async_session() as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(title="Corvex API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT 1"))
    value = result.scalar()
    return {"status": "ok", "db": "connected" if value == 1 else "unexpected_result"}


class QueryRequest(BaseModel):
    question: str
    k: int = 10

@app.post("/query")
async def query(req: QueryRequest, db: AsyncSession = Depends(get_db)):
    results = await hybrid_search(db, req.question, k=req.k)
    return {"question": req.question, "results": results}


@app.get("/query/stream")
async def query_stream(question: str, k: int = 10, db: AsyncSession = Depends(get_db)):
    retrieval_start = time.monotonic()
    chunks = await hybrid_search(db, question, k=k)
    retrieval_time = time.monotonic() - retrieval_start

    async def event_generator():
        generation_start = time.monotonic()
        usage_tracker = {}
        full_answer = ""
        async for token in generate_answer(question, chunks, usage_tracker=usage_tracker):
            full_answer += token
            payload = json.dumps({"token": token})
            yield f"event: token\ndata: {payload}\n\n"
        generation_time = time.monotonic() - generation_start

        citations = extract_citations(full_answer, chunks)
        payload = json.dumps({"citations": citations})
        yield f"event: citations\ndata: {payload}\n\n"

        logger.info(
            "Query complete | retrieval: %.2fs | generation: %.2fs | "
            "prompt_tokens: %d | completion_tokens: %d",
            retrieval_time,
            generation_time,
            usage_tracker.get("prompt_tokens", 0),
            usage_tracker.get("completion_tokens", 0),
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")