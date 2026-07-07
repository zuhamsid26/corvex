"""
Throwaway manual test script for keyword_search — not part of the app,
just for sanity-checking full-text search by hand. Delete before final cleanup.
"""

import asyncio

from main import async_session
from retrieval import keyword_search


async def main():
    questions = [
        "HTTPAdapter",
        "send_request",
        "connection pooling",
    ]

    async with async_session() as db:
        for q in questions:
            print(f"\n=== Keyword query: {q} ===")
            results = await keyword_search(db, q, k=5)
            if not results:
                print("  (no matches)")
            for r in results:
                print(f"  [{r['rank']:.4f}] {r['filepath']} :: {r['symbol_name']} ({r['symbol_type']})")


if __name__ == "__main__":
    asyncio.run(main())