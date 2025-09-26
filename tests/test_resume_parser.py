"""
Unittest suite for the resume parser.

These tests verify that the heuristic resume parser operates as
expected on a sample PDF, and that optional parsing paths (MinerU
and LLM) fall back gracefully when dependencies or API keys are not
available.  They avoid external network calls and rely solely on
built‑in Python libraries.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jobflow.resume.parse_resume import parse_resume


class TestResumeParser(unittest.TestCase):
    """Test cases for the resume parsing functions."""

    @classmethod
    def setUpClass(cls) -> None:
        # Determine the path to the sample PDF.  It resides at the project
        # root under the filename ``jtarriela_resume[sp].pdf``.
        repo_root = Path(__file__).resolve().parents[2]
        cls.pdf_path = repo_root / "jtarriela_resume[sp].pdf"
        if not cls.pdf_path.exists():
            raise unittest.SkipTest(f"Sample resume PDF {cls.pdf_path} not found")

    def test_parse_resume_regex(self) -> None:
        """Parse the resume PDF using the regex parser and validate fields."""
        resume = parse_resume(str(self.pdf_path), use_llm=False, use_mineru=False)
        # Full name should not be empty
        self.assertTrue(resume.full_name, "resume.full_name should not be empty")
        # Email field should exist
        self.assertIn("email", resume.contact)
        # Roles list should be present (may be empty if heuristics do not detect roles)
        self.assertIsInstance(resume.roles, list, "resume.roles should be a list")
        # Parsing method should be 'regex'
        self.assertEqual(resume.parsing_method, "regex")

    def test_parse_resume_mineru_fallback(self) -> None:
        """Simulate MinerU parsing when the package is unavailable."""
        # Patch the mineru import inside the module to None
        import importlib
        mod = importlib.import_module("jobflow.resume.parse_resume")
        with mock.patch.object(mod, "mineru", None):
            resume = parse_resume(str(self.pdf_path), use_llm=False, use_mineru=True)
        # Ensure fallback still produces a full_name
        self.assertTrue(resume.full_name)
        # Parsing method should remain 'regex'
        self.assertEqual(resume.parsing_method, "regex")

    def test_parse_resume_placeholder_llm(self) -> None:
        """Ensure that enabling LLM with a placeholder provider falls back."""
        # Clear any LLM API key environment variables and set provider to placeholder
        # Set API key environment variables to empty strings to ensure the
        # placeholder provider is chosen.  mock.patch.dict cannot set
        # environment variables to None because os.putenv expects str.
        env = {
            "OPENAI_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "LLM_PROVIDER": "placeholder",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            resume = parse_resume(str(self.pdf_path), use_llm=True, use_mineru=False)
        # When the provider is placeholder, fallback occurs and parsing_method is 'regex'
        self.assertEqual(resume.parsing_method, "regex")
        self.assertTrue(resume.full_name)


if __name__ == "__main__":
    unittest.main()