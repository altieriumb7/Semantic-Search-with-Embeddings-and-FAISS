import numpy as np

from src.config import INDEX_DIR
from src.embeddings import load_cached_embeddings, save_cached_embeddings


def test_embedding_cache_preserves_shape():
    texts = ["reset password", "download invoice"]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    embeddings_path = INDEX_DIR / "test_embeddings_shape.npy"
    meta_path = INDEX_DIR / "test_embeddings_shape_meta.json"

    try:
        save_cached_embeddings(embeddings, texts, "test-model", embeddings_path, meta_path)
        loaded = load_cached_embeddings(texts, "test-model", embeddings_path, meta_path)

        assert loaded is not None
        assert loaded.shape == (2, 3)
        np.testing.assert_array_equal(loaded, embeddings)
    finally:
        embeddings_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)


def test_embedding_cache_misses_on_text_change():
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    embeddings_path = INDEX_DIR / "test_embeddings_miss.npy"
    meta_path = INDEX_DIR / "test_embeddings_miss_meta.json"

    try:
        save_cached_embeddings(embeddings, ["original"], "test-model", embeddings_path, meta_path)

        assert load_cached_embeddings(["changed"], "test-model", embeddings_path, meta_path) is None
    finally:
        embeddings_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
