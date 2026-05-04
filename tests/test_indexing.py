import numpy as np
import pytest

from src.indexing import build_faiss_index


def test_build_faiss_index_adds_all_vectors():
    pytest.importorskip("faiss")
    embeddings = np.eye(4, dtype=np.float32)

    index = build_faiss_index(embeddings)

    assert index.ntotal == 4
    scores, indices = index.search(embeddings[:1], 2)
    assert indices.tolist()[0][0] == 0
    assert scores.tolist()[0][0] == pytest.approx(1.0)

