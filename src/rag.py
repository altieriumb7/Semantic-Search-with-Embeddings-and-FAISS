from __future__ import annotations

from typing import Sequence

from src.retrieval import SearchResult


def extractive_answer(query: str, results: Sequence[SearchResult], max_sentences: int = 3) -> str:
    """Optional no-API answer draft from retrieved chunks; not used for retrieval metrics."""
    if not results:
        return "No retrieved context was available to draft an answer."
    sentences: list[str] = []
    for result in results:
        first_sentence = result.chunk.text.split(".")[0].strip()
        if first_sentence:
            sentences.append(first_sentence + ".")
        if len(sentences) >= max_sentences:
            break
    return " ".join(sentences)

