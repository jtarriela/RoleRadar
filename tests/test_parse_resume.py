"""Tests for the résumé parsing logic.

These tests exercise the `parse_resume` function in both heuristic and LLM
modes.  Because real LLM providers are not available during testing, we
monkeypatch the provider factory to return a dummy provider when testing
the LLM pathway.  This ensures that the fallback heuristic parser is
invoked correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest  # type: ignore

from jobflow.resume.parse_resume import parse_resume


def test_parse_resume_heuristic(tmp_path: Path) -> None:
    """Ensure the heuristic parser extracts expected fields."""
    sample_text = (
        "Jane Doe\n"
        "Email: jane@example.com\n"
        "Phone: 123-456-7890\n"
        "Skills: Python, Machine Learning\n"
        "BS in Computer Science, 2018\n"
    )
    file_path = tmp_path / "resume.txt"
    file_path.write_text(sample_text, encoding="utf-8")
    resume = parse_resume(str(file_path), use_llm=False)
    assert resume.full_name == "Jane Doe"
    assert "Python" in resume.skills
    # Email extraction should find the email address
    assert resume.contact["email"] == "jane@example.com"
    # Phone extraction should contain the digits
    assert "123" in resume.contact["phone"]
    # Education extraction should capture the degree and year
    assert resume.education[0]["degree"] == "BS"
    assert resume.education[0]["year"] == 2018


def test_parse_resume_llm_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure that when the LLM provider is a placeholder, the heuristic parser is used."""
    # Monkeypatch the provider factory to return a placeholder provider
    import jobflow.rank.llm_providers as providers

    class DummyProvider:
        """A dummy provider whose class name triggers the placeholder logic."""

        # Set the class name directly
        __name__ = "PlaceholderProvider"

    def dummy_get_default_provider() -> DummyProvider:
        return DummyProvider()  # type: ignore[return-value]

    monkeypatch.setattr(providers, "get_default_provider", dummy_get_default_provider)

    sample_text = "John Smith\nEmail: john@example.com\nSkills: Data Science, Python"
    file_path = tmp_path / "resume.txt"
    file_path.write_text(sample_text, encoding="utf-8")
    # Even though use_llm=True, the dummy provider will cause fallback to heuristics
    resume = parse_resume(str(file_path), use_llm=True)
    assert resume.full_name == "John Smith"
    assert "Python" in resume.skills