"""
Throwaway manual test script for hybrid_search (RRF merge) — sanity-check
that fusion actually behaves sensibly compared to either leg alone.
"""

import asyncio

from main import async_session
from retrieval import hybrid_search


async def main():
    questions = [
        "How does connection pooling work?",
        "HTTPAdapter",
        "How are redirects handled?",
    ]

    async with async_session() as db:
        for q in questions:
            print(f"\n=== Hybrid query: {q} ===")
            results = await hybrid_search(db, q, k=5)
            for r in results:
                print(f"  [rrf={r['rrf_score']:.4f}] {r['filepath']} :: {r['symbol_name']} ({r['symbol_type']})")


if __name__ == "__main__":
    asyncio.run(main())