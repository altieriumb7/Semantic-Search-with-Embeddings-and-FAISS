from src.config import DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from src.evaluate_retrieval import evaluation_input_fingerprint, report_is_current


def test_report_is_current_checks_model_k_and_fingerprint():
    report = {
        "k": DEFAULT_TOP_K,
        "model_name": DEFAULT_MODEL_NAME,
        "input_fingerprint": evaluation_input_fingerprint(DEFAULT_MODEL_NAME),
    }

    assert report_is_current(report, DEFAULT_MODEL_NAME, DEFAULT_TOP_K)
    assert not report_is_current(report, "other-model", DEFAULT_TOP_K)
