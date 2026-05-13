from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import DEFAULT_MODEL_NAME, EMBEDDING_META_PATH, EMBEDDINGS_PATH, MODEL_CACHE_DIR


def cached_model_snapshot_path(model_name: str, cache_dir: Path | None = None) -> Path | None:
    cache_root = cache_dir or MODEL_CACHE_DIR
    repo_cache = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots = repo_cache / "snapshots"
    if not snapshots.exists():
        return None

    ref_path = repo_cache / "refs" / "main"
    if ref_path.exists():
        snapshot = snapshots / ref_path.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return snapshot

    snapshot_dirs = sorted((path for path in snapshots.iterdir() if path.is_dir()), reverse=True)
    return snapshot_dirs[0] if snapshot_dirs else None


class SentenceTransformerEmbedder:
    """Thin wrapper around SentenceTransformers for normalized dense embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Path | None = None,
        local_files_only: bool = False,
    ):
        from sentence_transformers import SentenceTransformer

        model_cache = cache_dir or MODEL_CACHE_DIR
        model_identifier = model_name
        if local_files_only:
            snapshot_path = cached_model_snapshot_path(model_name, model_cache)
            if snapshot_path is None:
                raise FileNotFoundError(
                    f"Model {model_name!r} is not cached under {model_cache}. "
                    "Run `python -m src.build_index` once with network access, or rerun search with `--allow-download`."
                )
            model_identifier = str(snapshot_path)

        self.model_name = model_name
        self.model = SentenceTransformer(
            model_identifier,
            cache_folder=str(model_cache),
            local_files_only=local_files_only,
        )

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )
        return np.asarray(embeddings, dtype=np.float32)


def embedding_fingerprint(texts: Sequence[str], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for text in texts:
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def load_cached_embeddings(
    texts: Sequence[str],
    model_name: str,
    embeddings_path: Path = EMBEDDINGS_PATH,
    meta_path: Path = EMBEDDING_META_PATH,
) -> np.ndarray | None:
    if not embeddings_path.exists() or not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = embedding_fingerprint(texts, model_name)
    if metadata.get("fingerprint") != expected:
        return None
    embeddings = np.load(embeddings_path)
    return np.asarray(embeddings, dtype=np.float32)


def save_cached_embeddings(
    embeddings: np.ndarray,
    texts: Sequence[str],
    model_name: str,
    embeddings_path: Path = EMBEDDINGS_PATH,
    meta_path: Path = EMBEDDING_META_PATH,
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, np.asarray(embeddings, dtype=np.float32))
    metadata = {
        "model_name": model_name,
        "fingerprint": embedding_fingerprint(texts, model_name),
        "num_embeddings": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def get_or_create_embeddings(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    force_rebuild: bool = False,
    batch_size: int = 32,
) -> np.ndarray:
    if not force_rebuild:
        cached = load_cached_embeddings(texts, model_name)
        if cached is not None and cached_model_snapshot_path(model_name) is not None:
            return cached

    embedder = SentenceTransformerEmbedder(model_name=model_name)
    embeddings = embedder.encode(texts, batch_size=batch_size)
    save_cached_embeddings(embeddings, texts, model_name)
    return embeddings
