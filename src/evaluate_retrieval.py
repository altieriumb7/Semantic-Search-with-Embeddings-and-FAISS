from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K, EVALUATION_REPORT_PATH
from src.data_loader import EvaluationQuery, load_evaluation_queries
from src.metrics import aggregate_metrics, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, success_at_1
from src.retrieval import SearchResult, SemanticSearcher, TfidfSearcher, result_doc_ids


def evaluate_results(results: Sequence[SearchResult], relevant_doc_ids: Sequence[str], k: int) -> dict:
    retrieved_doc_ids = result_doc_ids(results)[:k]
    relevant = set(relevant_doc_ids)
    return {
        "retrieved_doc_ids": retrieved_doc_ids,
        "precision_at_k": precision_at_k(retrieved_doc_ids, relevant, k),
        "recall_at_k": recall_at_k(retrieved_doc_ids, relevant, k),
        "mrr": reciprocal_rank(retrieved_doc_ids, relevant),
        "success_at_1": success_at_1(retrieved_doc_ids, relevant),
        "ndcg_at_k": ndcg_at_k(retrieved_doc_ids, relevant, k),
    }


def evaluate_searcher(searcher, queries: Sequence[EvaluationQuery], k: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for query in queries:
        # Retrieve more chunks than requested so duplicate chunks from one document do not crowd out doc-level scoring.
        results = searcher.search(query.query, top_k=max(k * 3, k))
        metrics = evaluate_results(results, query.relevant_doc_ids, k)
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "relevant_doc_ids": list(query.relevant_doc_ids),
                **metrics,
            }
        )
    return rows, aggregate_metrics(rows)


def build_qualitative_examples(semantic_rows: Sequence[dict], tfidf_rows: Sequence[dict], limit: int = 5) -> list[dict]:
    examples = []
    for semantic, tfidf in zip(semantic_rows, tfidf_rows):
        if len(examples) >= limit:
            break
        examples.append(
            {
                "query_id": semantic["query_id"],
                "query": semantic["query"],
                "relevant_doc_ids": semantic["relevant_doc_ids"],
                "semantic_top_docs": semantic["retrieved_doc_ids"][:3],
                "tfidf_top_docs": tfidf["retrieved_doc_ids"][:3],
            }
        )
    return examples


def evaluate_retrieval(
    k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL_NAME,
    report_path: Path = EVALUATION_REPORT_PATH,
) -> dict:
    queries = load_evaluation_queries()
    semantic_rows, semantic_summary = evaluate_searcher(
        SemanticSearcher(model_name=model_name, local_files_only=True),
        queries,
        k,
    )
    tfidf_rows, tfidf_summary = evaluate_searcher(TfidfSearcher(), queries, k)

    report = {
        "dataset": {
            "type": "synthetic",
            "description": "Synthetic support/product documentation with hand-labeled relevant document IDs.",
            "num_queries": len(queries),
        },
        "k": k,
        "metrics": {
            "semantic": semantic_summary,
            "tfidf": tfidf_summary,
        },
        "per_query": {
            "semantic": semantic_rows,
            "tfidf": tfidf_rows,
        },
        "qualitative_examples": build_qualitative_examples(semantic_rows, tfidf_rows),
        "notes": [
            "Metrics are computed on document IDs, while retrieval is performed over chunks.",
            "RAG-style answer generation is intentionally excluded from retrieval metrics.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic search against a TF-IDF baseline.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_retrieval(k=args.top_k, model_name=args.model_name)
    print(f"Evaluation report written to {EVALUATION_REPORT_PATH}")
    print(json.dumps(report["metrics"], indent=2))
    print("\nQualitative examples:")
    for example in report["qualitative_examples"]:
        print(
            f"- {example['query']} | relevant={example['relevant_doc_ids']} "
            f"| semantic={example['semantic_top_docs']} | tfidf={example['tfidf_top_docs']}"
        )


if __name__ == "__main__":
    main()
