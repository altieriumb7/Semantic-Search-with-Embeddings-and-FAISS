from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from src.data_loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    source: str
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict) -> "Chunk":
        return cls(
            chunk_id=str(record["chunk_id"]),
            doc_id=str(record["doc_id"]),
            title=str(record["title"]),
            category=str(record["category"]),
            source=str(record.get("source", "unknown")),
            chunk_index=int(record["chunk_index"]),
            text=str(record["text"]),
        )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size_words: int = 80, overlap_words: int = 20) -> list[str]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = normalize_text(text).split()
    if not words:
        return []
    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = chunk_size_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def chunk_document(
    document: Document,
    chunk_size_words: int = 80,
    overlap_words: int = 20,
) -> list[Chunk]:
    chunks = []
    for index, text in enumerate(chunk_text(document.text, chunk_size_words, overlap_words)):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::chunk_{index}",
                doc_id=document.doc_id,
                title=document.title,
                category=document.category,
                source=document.source,
                chunk_index=index,
                text=text,
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size_words: int = 80,
    overlap_words: int = 20,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size_words, overlap_words))
    if not chunks:
        raise ValueError("No chunks were produced from the input documents")
    return chunks

