"""CLI: python cli.py "question" --files a.pdf b.txt"""
from __future__ import annotations

import sys
from rag_core import run_pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python cli.py "your question" [--files a.pdf b.txt ...]')
        sys.exit(1)

    args = sys.argv[1:]
    if "--files" in args:
        split = args.index("--files")
        query_parts, file_paths = args[:split], args[split + 1:]
    else:
        query_parts, file_paths = args, []

    result = run_pipeline(" ".join(query_parts), file_paths)

    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for u in result["sources"]:
        print(f"- {u}")
    for f in result["files"]:
        print(f"- {f} (local file)")
