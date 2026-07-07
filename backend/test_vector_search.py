"""
Throwaway manual test script for vector_search — not part of the app,
just for sanity-checking retrieval quality by hand. Delete before final cleanup.
"""

import asyncio

from main import async_session
from retrieval import vector_search


async def main():
    questions = [
        "How does connection pooling work?",
        "What does the Session class do?",
        "How are redirects handled?",
    ]

    async with async_session() as db:
        for q in questions:
            print(f"\n=== Query: {q} ===")
            results = await vector_search(db, q, k=5)
            for r in results:
                print(f"  [{r['distance']:.4f}] {r['filepath']} :: {r['symbol_name']} ({r['symbol_type']})")


if __name__ == "__main__":
    asyncio.run(main())