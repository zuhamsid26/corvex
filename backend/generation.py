"""
Generation logic for Corvex: builds the grounding prompt from retrieved chunks
and calls Gemini to produce a cited, grounded answer.
"""

import logging
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a code assistant that answers questions about a codebase using ONLY the provided context chunks below.

Rules:
- Answer only using information present in the provided context. Do not use general knowledge about this library or any other source.
- If the answer is not fully contained in the provided context, say so explicitly rather than guessing.
- For every claim you make, cite the specific file and symbol it came from, in the format: (file: <filepath>, symbol: <symbol_name>).
- IMPORTANT: The <symbol_name> in your citation must be EXACTLY the symbol name shown in the context label above each chunk (e.g. "HTTPAdapter"), never a more specific method or attribute name within it (e.g. NOT "HTTPAdapter.init_poolmanager"). If you're describing something that happens inside a specific method of a class, still cite the containing class/function's symbol name exactly as labeled, since that's the actual retrievable unit.
- If you use the get_full_file tool to see a complete file, that file's content is NOT pre-labeled with symbol names the way the initial context is. When citing information that came specifically from a get_full_file result rather than the originally provided labeled context, use symbol: (module-level) if the information is outside any class/function body, or state clearly that the specific symbol is unknown rather than reusing a symbol name from the original labeled context that doesn't actually contain that information.
"""


CITATION_PATTERN = re.compile(
    r"\(file:\s*([^,]+?),\s*symbol:\s*([^)]+?)\)"
)

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GENERATION_MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=GEMINI_API_KEY)

# Anchor directory for resolving chunk filepaths stored during ingestion
# (e.g. "../corvex_data/requests/src/requests/adapters.py"). These paths
# are relative to backend/, where main.py and this module both live —
# confirmed against the actual ingested files before building this tool.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string for the prompt,
    each labeled with its file/symbol so the model can cite it correctly.
    """
    parts = []
    for chunk in chunks:
        header = f"--- {chunk['filepath']} :: {chunk.get('symbol_name') or '(module-level)'} ---"
        parts.append(f"{header}\n{chunk['code_text']}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the full user-turn prompt: context + question."""
    context_block = build_context_block(chunks)
    return f"""Context:

{context_block}

Question: {question}"""

def get_full_file(filepath: str) -> str:
    """Read and return the full contents of a source file from the ingested
    repo, given the same filepath format stored in code_chunks. This is the
    single scoped tool the model can call when a retrieved chunk alone isn't
    enough context (e.g. it needs to see an import at the top of the file,
    or a sibling function).
    """
    resolved_path = os.path.normpath(os.path.join(BACKEND_DIR, filepath))

    # Guard: never allow the tool to read outside the ingested corpus_data
    # directory, regardless of what path the model requests. This is the
    # containment boundary — the model only ever gets read access to files
    # that were already part of the ingested, public corpus.
    corvex_data_root = os.path.normpath(os.path.join(BACKEND_DIR, "..", "corvex_data"))
    if not resolved_path.startswith(corvex_data_root):
        return f"Error: access to '{filepath}' is outside the allowed corpus directory."

    if not os.path.isfile(resolved_path):
        return f"Error: file '{filepath}' not found."

    with open(resolved_path, "r", encoding="utf-8") as f:
        return f.read()


GET_FULL_FILE_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_full_file",
            description="Reads and returns the full contents of a source file from the ingested repository, given its filepath (as shown in the context labels).",
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The filepath of the file to read, exactly as shown in the context labels above each chunk.",
                    }
                },
                "required": ["filepath"],
            },
        )
    ]
)


async def generate_answer(question: str, chunks: list[dict], usage_tracker: dict | None = None):
    """Streams a grounded answer for the given question, using the retrieved
    chunks as context. Yields text tokens as they arrive. If the model
    requests get_full_file mid-stream, executes it and continues generating
    with the additional context — capped at a few tool-call rounds to
    prevent runaway loops.

    If usage_tracker is provided (a plain dict), it's populated in-place
    with cumulative token usage across all Gemini calls made during this
    turn (including any tool-call round-trips) — the caller can inspect it
    after the generator is fully consumed.
    """
    prompt = build_prompt(question, chunks)
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[GET_FULL_FILE_TOOL],
    )

    max_tool_rounds = 3
    for _ in range(max_tool_rounds):
        function_call = None
        model_content = None

        stream = client.models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=contents,
            config=config,
        )

        last_usage = None

        for chunk in stream:
            if chunk.usage_metadata:
                # usage_metadata is cumulative WITHIN a single stream — each
                # chunk reports the running total so far, not a delta. So we
                # only keep the most recent one seen, and apply it once
                # after the stream ends (not accumulate every chunk).
                last_usage = chunk.usage_metadata

            if chunk.function_calls:
                function_call = chunk.function_calls[0]
                model_content = chunk.candidates[0].content
                break
            if chunk.text:
                yield chunk.text

        if usage_tracker is not None and last_usage is not None:
            # Sum across calls (e.g. across tool-call rounds), since each
            # round is a genuinely separate Gemini request with its own
            # cumulative total.
            usage_tracker["prompt_tokens"] = usage_tracker.get("prompt_tokens", 0) + (last_usage.prompt_token_count or 0)
            usage_tracker["completion_tokens"] = usage_tracker.get("completion_tokens", 0) + (last_usage.candidates_token_count or 0)

        if function_call is None:
            return

        if function_call.name == "get_full_file":
            logger.info("Tool call: get_full_file(filepath=%r)", function_call.args["filepath"])
            result = get_full_file(function_call.args["filepath"])
        else:
            result = f"Error: unknown tool '{function_call.name}'"

        contents.append(model_content)
        contents.append(
            types.Content(role="user", parts=[types.Part.from_function_response(
                name=function_call.name, response={"result": result}
            )])
        )

    yield "\n\n[Note: reached max tool-call rounds without a final answer.]"


def extract_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    """Parse (file: ..., symbol: ...) citations out of the generated answer
    text. Returns an empty list if no citations were found — this is
    correct behavior both when the model explicitly found nothing relevant
    (e.g. "not in context") and, less commonly, if citation-format parsing
    failed on a substantive answer. We deliberately do NOT fall back to
    returning all retrieved chunks in either case: showing chunks the model
    never actually cited would mislead the user into trusting sources that
    weren't really used. A parsing failure on a substantive answer is logged
    as a warning so it's visible in instrumentation, without polluting the
    user-facing citation list with guesses.
    """
    matches = CITATION_PATTERN.findall(answer_text)

    if not matches:
        # Heuristic: a long, substantive answer with zero parsed citations
        # is more likely a format-parsing miss than a genuine "nothing
        # relevant" case (which tends to be a short, explicit statement).
        # This is purely diagnostic — it does not change what's returned.
        if len(answer_text.strip()) > 200:
            logger.warning(
                "No citations parsed from a substantive answer (%d chars) — "
                "possible citation format drift.",
                len(answer_text),
            )
        return []

    seen = set()
    citations = []
    for filepath, symbol_name in matches:
        filepath = filepath.strip()
        symbol_name = symbol_name.strip()
        key = (filepath, symbol_name)
        if key in seen:
            continue
        seen.add(key)

        matching_chunk = next(
            (c for c in chunks if c["filepath"] == filepath and (c.get("symbol_name") or "(module-level)") == symbol_name),
            None,
        )
        citations.append({
            "filepath": filepath,
            "symbol_name": symbol_name,
            "start_line": matching_chunk.get("start_line") if matching_chunk else None,
            "end_line": matching_chunk.get("end_line") if matching_chunk else None,
        })

    return citations