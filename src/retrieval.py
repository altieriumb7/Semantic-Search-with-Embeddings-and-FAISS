from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.chunking import Chunk
from src.config import DEFAULT_MODEL_NAME
from src.embeddings import SentenceTransformerEmbedder
from src.indexing import load_chunks, load_faiss_index, load_tfidf_artifacts
from src.tfidf import rank_tfidf


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk: Chunk
    method: str


class SemanticSearcher:
    def __init__(
        self,
        chunks: Sequence[Chunk] | None = None,
        faiss_index=None,
        embedder: SentenceTransformerEmbedder | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        local_files_only: bool = True,
    ):
        self.chunks = list(chunks) if chunks is not None else load_chunks()
        self.index = faiss_index if faiss_index is not None else load_faiss_index()
        self.embedder = (
            embedder
            if embedder is not None
            else SentenceTransformerEmbedder(model_name=model_name, local_files_only=local_files_only)
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_embedding = self.embedder.encode([query])
        scores, indices = self.index.search(np.asarray(query_embedding, dtype=np.float32), top_k)
        results: list[SearchResult] = []
        for rank, (index, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if index < 0:
                continue
            results.append(SearchResult(rank=rank, score=float(score), chunk=self.chunks[int(index)], method="semantic"))
        return results


class TfidfSearcher:
    def __init__(self, chunks: Sequence[Chunk] | None = None, vectorizer=None, matrix=None):
        self.chunks = list(chunks) if chunks is not None else load_chunks()
        if vectorizer is None or matrix is None:
            vectorizer, matrix = load_tfidf_artifacts()
        self.vectorizer = vectorizer
        self.matrix = matrix

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        ranked = rank_tfidf(query, self.vectorizer, self.matrix, top_k)
        return [
            SearchResult(rank=rank, score=score, chunk=self.chunks[index], method="tfidf")
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


def result_doc_ids(results: Sequence[SearchResult]) -> list[str]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    for result in results:
        if result.chunk.doc_id not in seen:
            seen.add(result.chunk.doc_id)
            doc_ids.append(result.chunk.doc_id)
    return doc_ids
