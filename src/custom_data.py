from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable

from src.data_loader import Document, EvaluationQuery

MAX_UPLOAD_BYTES = 1_000_000
MAX_CUSTOM_DOCUMENTS = 25
MAX_CUSTOM_TEXT_CHARS = 50_000


def _decode_upload(content: bytes) -> str:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Upload is too large. Limit is {MAX_UPLOAD_BYTES // 1_000_000} MB.")
    return content.decode("utf-8-sig")


def _read_jsonl_records(text: str) -> list[dict]:
    records: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_number}") from exc
    return records


def _read_csv_records(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV upload must include a header row.")
    return [dict(row) for row in reader]


def _records_from_upload(filename: str, content: bytes) -> list[dict]:
    text = _decode_upload(content)
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "jsonl":
        return _read_jsonl_records(text)
    if suffix == "csv":
        return _read_csv_records(text)
    raise ValueError("Upload must be a .jsonl or .csv file.")


def _clean_text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required field: {field}")
    return text


def documents_from_upload(filename: str, content: bytes) -> list[Document]:
    records = _records_from_upload(filename, content)
    if not records:
        raise ValueError("Upload did not contain any documents.")
    if len(records) > MAX_CUSTOM_DOCUMENTS:
        raise ValueError(f"Too many documents. Limit is {MAX_CUSTOM_DOCUMENTS}.")

    documents: list[Document] = []
    seen_ids: set[str] = set()
    total_chars = 0
    for index, record in enumerate(records, start=1):
        text = _clean_text(record.get("text"), "text")
        total_chars += len(text)
        if total_chars > MAX_CUSTOM_TEXT_CHARS:
            raise ValueError(f"Uploaded text is too large. Limit is {MAX_CUSTOM_TEXT_CHARS} characters.")
        doc_id = str(record.get("doc_id") or f"custom_doc_{index}").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", doc_id):
            raise ValueError(f"Invalid doc_id {doc_id!r}; use letters, numbers, dots, dashes, underscores, or colons.")
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        seen_ids.add(doc_id)
        documents.append(
            Document(
                doc_id=doc_id,
                title=str(record.get("title") or doc_id).strip(),
                category=str(record.get("category") or "custom").strip(),
                source=str(record.get("source") or "visitor_upload").strip(),
                text=text,
            )
        )
    return documents


def _split_relevant_ids(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        ids = [str(item).strip() for item in value]
    else:
        ids = [part.strip() for part in re.split(r"[;,]", str(value or ""))]
    ids = [doc_id for doc_id in ids if doc_id]
    if not ids:
        raise ValueError("Each evaluation query needs at least one relevant_doc_ids value.")
    return tuple(ids)


def evaluation_queries_from_upload(filename: str, content: bytes) -> list[EvaluationQuery]:
    records = _records_from_upload(filename, content)
    if not records:
        raise ValueError("Upload did not contain any evaluation queries.")
    queries: list[EvaluationQuery] = []
    for index, record in enumerate(records, start=1):
        query = _clean_text(record.get("query"), "query")
        queries.append(
            EvaluationQuery(
                query_id=str(record.get("query_id") or f"custom_query_{index}").strip(),
                query=query,
                relevant_doc_ids=_split_relevant_ids(record.get("relevant_doc_ids")),
            )
        )
    return queries


def document_payload(documents: Iterable[Document]) -> tuple[tuple[str, str, str, str, str], ...]:
    return tuple((doc.doc_id, doc.title, doc.category, doc.source, doc.text) for doc in documents)


def documents_from_payload(payload: Iterable[tuple[str, str, str, str, str]]) -> list[Document]:
    return [Document(doc_id=row[0], title=row[1], category=row[2], source=row[3], text=row[4]) for row in payload]
