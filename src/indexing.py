from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from src.chunking import Chunk
from src.config import CHUNKS_PATH, FAISS_INDEX_PATH, TFIDF_PATH


def save_chunks(chunks: Sequence[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk.to_dict(), ensure_ascii=True) + "\n")


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"Missing chunks file: {path}. Run `python -m src.build_index` first.")
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for vector indexing. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return faiss


def build_faiss_index(embeddings: np.ndarray):
    faiss = _import_faiss()
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def save_faiss_index(index, path: Path = FAISS_INDEX_PATH) -> None:
    faiss = _import_faiss()
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path = FAISS_INDEX_PATH):
    faiss = _import_faiss()
    if not path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {path}. Run `python -m src.build_index` first.")
    return faiss.read_index(str(path))


def save_tfidf_artifacts(vectorizer, matrix, path: Path = TFIDF_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, path)


def load_tfidf_artifacts(path: Path = TFIDF_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Missing TF-IDF artifacts: {path}. Run `python -m src.build_index` first.")
    artifacts = joblib.load(path)
    return artifacts["vectorizer"], artifacts["matrix"]

