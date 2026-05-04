from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from src.chunking import Chunk


def build_tfidf(chunks: Sequence[Chunk]):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    return vectorizer, matrix


def rank_tfidf(query: str, vectorizer, matrix, top_k: int = 5) -> list[tuple[int, float]]:
    if top_k <= 0:
        return []
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    order = scores.argsort()[::-1][:top_k]
    return [(int(index), float(scores[index])) for index in order]

