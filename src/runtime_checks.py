from __future__ import annotations

from pathlib import Path

from src.config import DEFAULT_MODEL_NAME, MODEL_CACHE_DIR, REQUIRED_SEARCH_ARTIFACTS
from src.embeddings import cached_model_snapshot_path


def missing_search_artifacts(model_name: str = DEFAULT_MODEL_NAME) -> list[Path]:
    missing = [path for path in REQUIRED_SEARCH_ARTIFACTS if not path.exists()]
    if cached_model_snapshot_path(model_name, MODEL_CACHE_DIR) is None:
        missing.append(MODEL_CACHE_DIR)
    return missing


def search_artifacts_ready(model_name: str = DEFAULT_MODEL_NAME) -> bool:
    return not missing_search_artifacts(model_name)
