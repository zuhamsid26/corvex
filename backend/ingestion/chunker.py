"""
AST-based code chunker.

Walks a Python codebase and extracts function/class-level chunks
(not blind text-splitting), preserving docstrings and metadata
needed for retrieval: filepath, symbol name/type, line range, source text.
"""

import ast
from logging import root
from fileinput import filename
import os
from dataclasses import dataclass, field


@dataclass
class Chunk:
    filepath: str
    symbol_name: str | None
    symbol_type: str  # 'function' | 'class' | 'module'
    start_line: int
    end_line: int
    code_text: str


def get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    """Extract the exact source text for a node using its line numbers."""
    start = node.lineno - 1  # ast line numbers are 1-indexed
    end = node.end_lineno  # end_lineno is inclusive, slicing is exclusive-friendly
    return "\n".join(source_lines[start:end])


def chunk_file(filepath: str) -> list[Chunk]:
    """
    Parse a single .py file and return a list of Chunks:
    one per top-level function, one per top-level class (methods stay
    embedded inside their class's chunk rather than being split out
    individually), and one 'module' chunk for any top-level code that
    isn't inside a function or class.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        print(f"  [skip] SyntaxError parsing {filepath}: {e}")
        return []

    chunks: list[Chunk] = []
    covered_lines: set[int] = set()  # track lines already claimed by a chunk

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            code_text = get_source_segment(source_lines, node)
            chunks.append(
                Chunk(
                    filepath=filepath,
                    symbol_name=node.name,
                    symbol_type="function",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    code_text=code_text,
                )
            )
            covered_lines.update(range(node.lineno, node.end_lineno + 1))

        elif isinstance(node, ast.ClassDef):
            code_text = get_source_segment(source_lines, node)
            chunks.append(
                Chunk(
                    filepath=filepath,
                    symbol_name=node.name,
                    symbol_type="class",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    code_text=code_text,
                )
            )
            covered_lines.update(range(node.lineno, node.end_lineno + 1))

    # Module-level code: anything not covered by a function/class chunk above
    # (imports, constants, top-level `if __name__ == "__main__":` blocks, etc.)
    all_lines = set(range(1, len(source_lines) + 1))
    leftover_lines = sorted(all_lines - covered_lines)

    if leftover_lines:
        # Only keep leftover lines that actually have non-blank content
        leftover_text_lines = [
            source_lines[i - 1] for i in leftover_lines
            if source_lines[i - 1].strip()
        ]
        if leftover_text_lines:
            chunks.append(
                Chunk(
                    filepath=filepath,
                    symbol_name=None,
                    symbol_type="module",
                    start_line=leftover_lines[0],
                    end_line=leftover_lines[-1],
                    code_text="\n".join(leftover_text_lines),
                )
            )

    return chunks


def chunk_repo(repo_path: str) -> list[Chunk]:
    """Walk a repo directory, chunk every .py file found."""
    all_chunks: list[Chunk] = []

    for root, dirs, files in os.walk(repo_path):
        # Skip common noise directories
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "venv", ".venv", "tests", "docs")]

        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename).replace("\\", "/")
                file_chunks = chunk_file(filepath)
                all_chunks.extend(file_chunks)

    return all_chunks


if __name__ == "__main__":
    # Quick manual sanity test on a small subset first
    repo_path = "../corvex_data/requests/src/requests"
    chunks = chunk_repo(repo_path)

    print(f"Total chunks extracted: {len(chunks)}\n")

    for c in chunks:
        print(f"[{c.symbol_type}] {c.symbol_name or '(module-level)'} - {c.filepath} (lines {c.start_line}-{c.end_line})")
        print("-" * 60)
        print(c.code_text[:200] + ("..." if len(c.code_text) > 200 else ""))
        print("=" * 60)