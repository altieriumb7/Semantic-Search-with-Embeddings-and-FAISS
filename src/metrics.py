from __future__ import annotations

import math
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


def success_at_1(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return 1.0 if retrieved_ids[0] in relevant_ids else 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        rel_i = 1.0 if doc_id in relevant_ids else 0.0
        if rel_i > 0:
            dcg += rel_i / math.log2(i + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def aggregate_metrics(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "success_at_1": 0.0, "ndcg_at_k": 0.0}
    return {
        "precision_at_k": sum(row["precision_at_k"] for row in rows) / len(rows),
        "recall_at_k": sum(row["recall_at_k"] for row in rows) / len(rows),
        "mrr": sum(row["mrr"] for row in rows) / len(rows),
        "success_at_1": sum(row["success_at_1"] for row in rows) / len(rows),
        "ndcg_at_k": sum(row["ndcg_at_k"] for row in rows) / len(rows),
    }
