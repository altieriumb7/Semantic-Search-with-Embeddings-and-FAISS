from src.runtime_checks import missing_search_artifacts


def test_missing_search_artifacts_reports_missing_model_cache(tmp_path):
    assert any(path.name == "model_cache" for path in missing_search_artifacts("missing/model"))
