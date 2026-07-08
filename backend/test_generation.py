"""
Throwaway manual test script for generate_answer — sanity-check streaming
and tool-call behavior by hand before wiring into the /query endpoint.
"""

import asyncio

from main import async_session
from retrieval import hybrid_search, vector_search
from generation import generate_answer



async def main():
    question = "In requests/adapters.py, which urllib3 exception classes are imported at the top of the file?"

    async with async_session() as db:
        # Deliberately fetch only the HTTPAdapter class chunk (k=1, vector-only),
        # which does NOT contain the import statements — forces the model to
        # either say the context is insufficient, or call get_full_file to check.
        chunks = await vector_search(db, "HTTPAdapter class definition", k=1)

    print(f"Question: {question}\n")
    print("Chunk(s) given as context:", [c['symbol_name'] for c in chunks])
    print("--- Streaming answer ---")
    async for token in generate_answer(question, chunks):
        print(token, end="", flush=True)
    print("\n--- End ---")


if __name__ == "__main__":
    asyncio.run(main())