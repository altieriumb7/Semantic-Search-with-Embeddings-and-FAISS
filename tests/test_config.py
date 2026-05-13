import pytest

from src.config import _positive_int_from_env


def test_positive_int_from_env_validates_values(monkeypatch):
    monkeypatch.setenv("DEFAULT_TOP_K", "3")
    assert _positive_int_from_env("DEFAULT_TOP_K", 5) == 3

    monkeypatch.setenv("DEFAULT_TOP_K", "0")
    with pytest.raises(ValueError):
        _positive_int_from_env("DEFAULT_TOP_K", 5)
