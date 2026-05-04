from __future__ import annotations

from typing import Sequence


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    hits = len({doc_id for doc_id in top_k if doc_id in relevant_ids})
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for index, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / index
    return 0.0


def aggregate_metrics(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    return {
        "precision_at_k": sum(row["precision_at_k"] for row in rows) / len(rows),
        "recall_at_k": sum(row["recall_at_k"] for row in rows) / len(rows),
        "mrr": sum(row["mrr"] for row in rows) / len(rows),
    }

