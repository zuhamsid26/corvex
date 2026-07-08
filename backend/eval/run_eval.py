"""
Evaluation harness for Corvex retrieval.

Runs every question in test_set.json through three retrieval modes
(vector-only, keyword-only, hybrid) and reports precision@k / recall@k
for each — the numbers used to justify why hybrid retrieval is the
right design choice, not just an assumption.

Note on precision@k here: since each question has exactly ONE correct
chunk (not multiple relevant ones), precision@k = recall@k / k for every
question — it's not an independent signal in this setup, just a rescaled
view of the same hit/miss outcome.
"""

import asyncio
import json
import os
import sys

# Add the backend/ directory to the path so we can import main.py and
# retrieval.py, which live one level up from this eval/ subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import async_session
from retrieval import vector_search, keyword_search, hybrid_search


def is_match(chunk: dict, expected_file: str, expected_symbol: str) -> bool:
    """A retrieved chunk counts as correct if its filepath ends with the
    expected_file (ingestion stored a longer relative path like
    '../corvex_data/requests/src/requests/models.py', not the short
    'requests/models.py' form used in the test set) AND its symbol_name
    matches exactly.
    """
    filepath = chunk.get("filepath", "")
    symbol = chunk.get("symbol_name")
    return filepath.endswith(expected_file) and symbol == expected_symbol


async def evaluate_mode(db, test_set: list[dict], search_fn, k: int) -> dict:
    """Run every question through one retrieval mode, return per-question
    hit/miss plus aggregate precision@k / recall@k.
    """
    hits = 0
    details = []

    for item in test_set:
        results = await search_fn(db, item["question"], k=k)
        found = any(
            is_match(r, item["expected_file"], item["expected_symbol"])
            for r in results
        )
        if found:
            hits += 1
        details.append({"question": item["question"], "found": found})

    n = len(test_set)
    recall_at_k = hits / n if n else 0.0
    precision_at_k = recall_at_k / k  # see module docstring: not independent here

    return {
        "hits": hits,
        "total": n,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "details": details,
    }


async def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    test_set_path = os.path.join(os.path.dirname(__file__), "test_set.json")
    with open(test_set_path) as f:
        test_set = json.load(f)

    print(f"Loaded {len(test_set)} test questions. Running eval at k={k}...\n")

    async with async_session() as db:
        vector_results = await evaluate_mode(db, test_set, vector_search, k)
        keyword_results = await evaluate_mode(db, test_set, keyword_search, k)
        hybrid_results = await evaluate_mode(db, test_set, hybrid_search, k)

    print(f"{'Mode':<15} {'Hits':<8} {'Recall@k':<12} {'Precision@k':<12}")
    print("-" * 50)
    for name, res in [
        ("Vector-only", vector_results),
        ("Keyword-only", keyword_results),
        ("Hybrid", hybrid_results),
    ]:
        print(
            f"{name:<15} {res['hits']}/{res['total']:<6} "
            f"{res['recall_at_k']:<12.2%} {res['precision_at_k']:<12.4f}"
        )

    # Print which specific questions failed for each mode — useful for
    # deciding what to tune (chunk size, k, RRF weighting) in the next step.
    for name, res in [
        ("Vector-only", vector_results),
        ("Keyword-only", keyword_results),
        ("Hybrid", hybrid_results),
    ]:
        misses = [d["question"] for d in res["details"] if not d["found"]]
        if misses:
            print(f"\n{name} missed:")
            for q in misses:
                print(f"  - {q}")


if __name__ == "__main__":
    asyncio.run(main())