from __future__ import annotations

import argparse

from src.chunking import chunk_documents
from src.config import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_MODEL_NAME,
    FAISS_INDEX_PATH,
    TFIDF_PATH,
)
from src.data_loader import load_documents
from src.embeddings import get_or_create_embeddings
from src.indexing import build_faiss_index, save_chunks, save_faiss_index, save_tfidf_artifacts
from src.tfidf import build_tfidf


def build_index(
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    force_rebuild: bool = False,
    batch_size: int = 32,
) -> dict:
    documents = load_documents()
    chunks = chunk_documents(documents, chunk_size_words, overlap_words)
    save_chunks(chunks)

    embeddings = get_or_create_embeddings(
        [chunk.text for chunk in chunks],
        model_name=model_name,
        force_rebuild=force_rebuild,
        batch_size=batch_size,
    )
    faiss_index = build_faiss_index(embeddings)
    save_faiss_index(faiss_index)

    vectorizer, matrix = build_tfidf(chunks)
    save_tfidf_artifacts(vectorizer, matrix)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_index": str(FAISS_INDEX_PATH),
        "tfidf_artifacts": str(TFIDF_PATH),
        "dataset": "synthetic support/product documentation",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS and TF-IDF search indexes.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunk-size-words", type=int, default=DEFAULT_CHUNK_SIZE_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_CHUNK_OVERLAP_WORDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore cached embeddings and rebuild artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_index(
        model_name=args.model_name,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
        force_rebuild=args.force_rebuild,
        batch_size=args.batch_size,
    )
    print("Built retrieval artifacts")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

