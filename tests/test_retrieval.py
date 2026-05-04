import numpy as np

from src.chunking import Chunk
from src.retrieval import SemanticSearcher, TfidfSearcher
from src.tfidf import build_tfidf


class FakeEmbedder:
    def encode(self, texts, batch_size=32):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "password" in text.lower() else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeIndex:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def search(self, query_embeddings, top_k):
        scores = query_embeddings @ self.matrix.T
        indices = np.argsort(scores[0])[::-1][:top_k]
        return scores[:, indices], indices.reshape(1, -1)


def sample_chunks():
    return [
        Chunk(
            chunk_id="password::chunk_0",
            doc_id="password",
            title="Password reset",
            category="account",
            source="test",
            chunk_index=0,
            text="Reset a forgotten password with a secure email link.",
        ),
        Chunk(
            chunk_id="invoice::chunk_0",
            doc_id="invoice",
            title="Invoices",
            category="billing",
            source="test",
            chunk_index=0,
            text="Download billing invoices and receipts from billing history.",
        ),
    ]


def test_semantic_retrieval_returns_matching_chunk():
    chunks = sample_chunks()
    index = FakeIndex([[1.0, 0.0], [0.0, 1.0]])
    searcher = SemanticSearcher(chunks=chunks, faiss_index=index, embedder=FakeEmbedder())

    results = searcher.search("password help", top_k=1)

    assert results[0].chunk.doc_id == "password"
    assert results[0].method == "semantic"


def test_tfidf_retrieval_returns_matching_chunk():
    chunks = sample_chunks()
    vectorizer, matrix = build_tfidf(chunks)
    searcher = TfidfSearcher(chunks=chunks, vectorizer=vectorizer, matrix=matrix)

    results = searcher.search("billing receipts", top_k=1)

    assert results[0].chunk.doc_id == "invoice"
    assert results[0].method == "tfidf"

