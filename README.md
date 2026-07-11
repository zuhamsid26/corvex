# Corvex

A retrieval system for natural-language Q&A over a codebase: point it at a repository, ask a question, and get a grounded, cited answer streamed back in real time — retrieved from actual functions and classes, not general LLM knowledge.

**Live demo:** [corvex-wheat.vercel.app](https://corvex-wheat.vercel.app)
**API:** [corvex-api.onrender.com](https://corvex-api.onrender.com/health)

> Note: the backend runs on Render's free tier, which spins down after periods of inactivity. The first request after idle time may take 30–60 seconds to respond while the instance wakes up.

---

## What This Is

Corvex is **not an ML project** — no model is trained anywhere in this system. Every AI-labeled component (embeddings, generation) is a hosted API call, integrated like any third-party service. The actual engineering lives in:

1. **AST-aware code chunking** — parsing code by structure (function/class boundaries), not blind text-splitting
2. **Hybrid retrieval** — pgvector cosine similarity + Postgres full-text search, merged via Reciprocal Rank Fusion
3. **A retrieval evaluation harness** — real precision/recall@k numbers, not "it seemed to work"
4. **Streaming generation with grounded citations** — SSE token streaming, answers traceable to file + function
5. **One scoped tool call** (`get_full_file`) — controlled agentic behavior, not an open-ended agent loop

The corpus for this demo is the [`requests`](https://github.com/psf/requests) library.

## Measured Retrieval Quality

A hand-built, 24-question evaluation set (spanning 5 files, mixing exact-identifier and conceptual questions) was run through all three retrieval modes at two values of *k*:

| Mode             | Recall@5  | Recall@10 |
|------------------|-----------|-----------|
| Vector-only      | 100%      | 100%      |
| Keyword-only     | 8.3%      | 8.3%      |
| **Hybrid (RRF)** | **95.8%** | **100%**  |

Keyword-only search performs poorly here specifically because the test set is dominated by natural-language, conceptual questions rather than bare identifiers — `ts_rank` scores on term frequency across the whole query, including filler words, so it doesn't reward semantic relevance the way embeddings do. This is the concrete, measured argument for why hybrid retrieval — not vector or keyword alone — is the right architecture for a codebase Q&A system where users ask real questions, not just search for symbol names.

`k=5` was chosen as the production default: both vector and hybrid retrieval already saturate at or near 100% recall at that window, and a smaller `k` means less context sent to the LLM per query — cheaper, faster, and more tightly grounded.

## Features

- **Hybrid search** — pgvector cosine similarity + Postgres full-text search (`websearch_to_tsquery` + `ts_rank`), merged via Reciprocal Rank Fusion with a widened candidate pool per leg before fusion
- **Streaming, cited answers** — Server-Sent Events stream tokens as they generate; a final `citations` event returns the specific file, symbol, and line range actually referenced in the answer
- **Scoped tool use** — the model can call `get_full_file` when a retrieved chunk alone isn't sufficient context (e.g. it needs an import statement or a sibling function), sandboxed to the ingested corpus directory only
- **Syntax-highlighted markdown rendering**, including code blocks in generated answers
- **Graceful degradation** — genuinely irrelevant questions ("what's the capital of France?") are correctly answered as out-of-context rather than hallucinated; parsing failures on citations fail safe to an empty list rather than showing misleading sources

## Tech Stack

**Backend**
- FastAPI (async), Server-Sent Events via `StreamingResponse`
- `asyncpg` + SQLAlchemy (async engine) for Postgres access
- PostgreSQL + `pgvector` extension, hosted on [Neon](https://neon.tech) (serverless Postgres, free tier)
- Python's built-in `ast` module for AST-based code chunking
- Google Gemini API — `gemini-embedding-001` for embeddings, `gemini-3.1-flash-lite` for generation (both free tier)
- `tenacity` for retry/backoff on embedding and generation calls

**Frontend**
- React (Vite) + Tailwind CSS v4
- `react-markdown` + `react-syntax-highlighter` for formatted, syntax-highlighted answers
- Native `EventSource` for SSE streaming

**Deployment**
- Backend: [Render](https://render.com) (free tier)
- Frontend: [Vercel](https://vercel.com) (Hobby tier)
- Cost: **$0** — both AI API usage and hosting stay within free-tier limits by design

## Architecture

```
┌──────────────┐        HTTPS/SSE        ┌──────────────┐
│   Frontend   │ ──────────────────────▶ │   Backend    │
│   (Vercel)   │ ◀────────────────────── │   (Render)   │
│  React+Vite  │      streamed JSON      │   FastAPI    │
└──────────────┘                         └──────┬───────┘
                                                │
                    ┌───────────────────────────┼─────────────────────────────┐
                    ▼                           ▼                             ▼
             ┌─────────────┐            ┌───────────────────┐          ┌──────────────┐
             │   Neon      │            │   Gemini API      │          │  Ingested    │
             │  Postgres   │            │ (embed + generate)│          │  repo files  │
             │  + pgvector │            └───────────────────┘          │  (get_full_  │
             └─────────────┘                                           │   file tool) │
                                                                       └──────────────┘
```

**Ingestion (offline, run once):** clone target repo → parse each `.py` file with `ast` → chunk by function/class boundary, keeping docstrings attached → embed each chunk via Gemini (concurrent requests, semaphore-capped, retry/backoff — Gemini's embedding endpoint doesn't support batch arrays) → insert into `code_chunks` with HNSW (vector) and GIN (full-text) indexes.

**Retrieval (`/query/stream`):** embed the question → run vector search (cosine similarity) and keyword search (full-text) in parallel candidate pools → merge via Reciprocal Rank Fusion, deduped by `chunk_id` → return top-*k*.

**Generation:** retrieved chunks are labeled with their file/symbol and assembled into a grounding prompt → Gemini streams a cited answer → if the model requests `get_full_file`, it's executed (sandboxed to the ingested corpus directory) and fed back in, capped at 3 tool-call rounds → citations are parsed from the final answer text and matched back to real chunk metadata.

## Project Structure

```
corvex/
├── backend/
│   ├── main.py                  # FastAPI app, /health, /query, /query/stream endpoints
│   ├── retrieval.py              # Vector search, keyword search, hybrid RRF merge
│   ├── generation.py             # Prompt construction, streaming generation, get_full_file tool, citation extraction
│   ├── requirements.txt
│   ├── ingestion/
│   │   ├── chunker.py             # AST-based code chunker
│   │   └── embed_and_store.py     # Gemini embedding + Postgres ingestion
│   └── eval/
│       ├── test_set.json          # 24 hand-verified questions with expected file/symbol
│       └── run_eval.py            # Precision/recall@k evaluation harness (vector/keyword/hybrid)
├── frontend/
│   └── src/
│       └── App.jsx                # Chat UI: SSE streaming, markdown rendering, citations panel
└── README.md
```

## API Overview

| Endpoint        | Method | Purpose                                                            |
|-----------------|--------|--------------------------------------------------------------------|
| `/health`       | GET    | Liveness check with a real DB round-trip                           |
| `/query`        | POST   | Retrieval-only — returns hybrid-merged chunks as JSON, no LLM call |
| `/query/stream` | GET    | Full pipeline — retrieval + streamed, cited generation via SSE     |

## Local Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash; venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

Create `backend/.env`:

```
DATABASE_URL=your_neon_connection_string
GEMINI_API_KEY=your_gemini_api_key
```

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

```bash
npm run dev
```

Frontend runs at `http://localhost:5173`.

## Known Limitations and Design Decisions

These are intentional scope boundaries for a portfolio project.

- **No authentication.** Anyone with the URL can query it.
- **No chat history persistence.** Conversation state lives in React's in-memory state only — refreshing the page clears it.
- **Single corpus, single language.** The demo is scoped to one Python repository (`requests`). Multi-repo ingestion and multi-language support (via tree-sitter) were explicitly deferred to keep retrieval-engineering depth as the focus rather than breadth of integrations.
- **Citation extraction depends on the model consistently following the prompted `(file: ..., symbol: ...)` format**, which it does not always do perfectly. The system fails safely — parsing misses return an empty citation list rather than a misleading one.
- **Free-tier hosting cold starts.** The backend (Render free tier) spins down after inactivity, causing a 30–60s delay on the first request after idle time.

## Acknowledgments

Built end-to-end as a solo project to learn and demonstrate information retrieval fundamentals (BM25-style full-text search, cosine similarity, reciprocal rank fusion) alongside practical LLM API integration (streaming, function-calling, prompt engineering for grounded citations).
