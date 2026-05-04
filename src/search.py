from __future__ import annotations

import argparse
import textwrap

from src.config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from src.retrieval import SemanticSearcher, TfidfSearcher


def format_results(title: str, results) -> str:
    lines = [title, "-" * len(title)]
    for result in results:
        preview = textwrap.shorten(result.chunk.text, width=180, placeholder="...")
        lines.append(
            f"{result.rank}. score={result.score:.4f} "
            f"doc_id={result.chunk.doc_id} title={result.chunk.title}\n   {preview}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the indexed document chunks.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--method", choices=["semantic", "tfidf", "both"], default="both")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow SentenceTransformers to contact Hugging Face if the model is not already cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method in {"semantic", "both"}:
        semantic = SemanticSearcher(model_name=args.model_name, local_files_only=not args.allow_download)
        print(format_results("Semantic search", semantic.search(args.query, args.top_k)))
    if args.method == "both":
        print()
    if args.method in {"tfidf", "both"}:
        tfidf = TfidfSearcher()
        print(format_results("TF-IDF baseline", tfidf.search(args.query, args.top_k)))


if __name__ == "__main__":
    main()
