from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import DOCUMENTS_PATH, EVALUATION_QUERIES_PATH


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    category: str
    source: str
    text: str


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def load_documents(path: Path = DOCUMENTS_PATH) -> list[Document]:
    documents: list[Document] = []
    for record in _read_jsonl(path):
        documents.append(
            Document(
                doc_id=str(record["doc_id"]),
                title=str(record["title"]),
                category=str(record["category"]),
                source=str(record.get("source", "unknown")),
                text=str(record["text"]),
            )
        )
    if not documents:
        raise ValueError(f"No documents found in {path}")
    return documents


def load_evaluation_queries(path: Path = EVALUATION_QUERIES_PATH) -> list[EvaluationQuery]:
    queries: list[EvaluationQuery] = []
    for record in _read_jsonl(path):
        relevant = tuple(str(doc_id) for doc_id in record["relevant_doc_ids"])
        queries.append(
            EvaluationQuery(
                query_id=str(record["query_id"]),
                query=str(record["query"]),
                relevant_doc_ids=relevant,
            )
        )
    if not queries:
        raise ValueError(f"No evaluation queries found in {path}")
    return queries

