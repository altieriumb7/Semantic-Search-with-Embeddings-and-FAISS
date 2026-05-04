from src.metrics import aggregate_metrics, precision_at_k, recall_at_k, reciprocal_rank


def test_ranking_metrics():
    retrieved = ["doc-c", "doc-a", "doc-b"]
    relevant = {"doc-a", "doc-d"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_aggregate_metrics():
    rows = [
        {"precision_at_k": 0.5, "recall_at_k": 1.0, "mrr": 1.0},
        {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0},
    ]

    assert aggregate_metrics(rows) == {"precision_at_k": 0.25, "recall_at_k": 0.5, "mrr": 0.5}

