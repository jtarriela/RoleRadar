"""
LLM provider abstractions.

This module defines a common interface for large language model (LLM)
providers used by RoleRadar to judge the relevance of a résumé to a job
posting.  Concrete implementations are provided for OpenAI and
Gemini (Google Generative AI) APIs.  A placeholder implementation
is used when no API keys are configured or the optional dependencies
are not installed.  Applications can select the provider via
environment variables or pass an instance of ``LLMProvider`` directly.

Compared to the previous version, the default Gemini model has been
updated to ``gemini‑1.5‑pro`` to align with the latest generation of
Google's Gemini models.  If you wish to use a different model, set
the ``GEMINI_MODEL`` or ``GOOGLE_MODEL`` environment variable
accordingly.
"""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def judge(self, resume_json: Dict[str, object], job_text: str, job_title: str, company: str) -> Tuple[float, List[str], List[str]]:
        """Judge the relevance of a résumé to a job posting.

        Args:
            resume_json: Parsed résumé JSON.
            job_text: Plain text description of the job.
            job_title: Job title string.
            company: Company name.

        Returns:
            Tuple of (score, reasons, missing_skills).
        """
        raise NotImplementedError


class PlaceholderProvider(LLMProvider):
    """Fallback provider that does not call any external API."""

    def judge(self, resume_json: Dict[str, object], job_text: str, job_title: str, company: str) -> Tuple[float, List[str], List[str]]:
        skills = set(resume_json.get("skills", []))
        # Determine missing must‑have skills: any resume skill not in job description
        missing = [s for s in skills if s.lower() not in job_text.lower()]
        reasons = [
            f"Basic role alignment with {job_title}",
            f"Company {company} and candidate skills overlap",
        ]
        relevance = 0.5  # neutral placeholder
        return relevance, reasons, missing


class OpenAIProvider(LLMProvider):
    """Provider that uses the OpenAI ChatCompletion API."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-3.5-turbo") -> None:
        try:
            import openai  # type: ignore
            self.openai = openai
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OpenAIProvider. Install it via pip."
            ) from exc
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        # Configure client
        self.openai.api_key = self.api_key

    def judge(self, resume_json: Dict[str, object], job_text: str, job_title: str, company: str) -> Tuple[float, List[str], List[str]]:
        # Build prompt using a simple scheme; production code should use
        # the structured prompt described in the README.  The model
        # returns a JSON with keys: score, reasons, missing.
        skills = ", ".join(resume_json.get("skills", []))
        prompt = (
            "You are a recruiting assistant. Given a candidate résumé and a job description, "
            "assess how well the candidate fits the role. Return a JSON object with keys "
            "'score' (0 to 1), 'reasons' (list of strings explaining the match), and 'missing' "
            "(list of required skills missing from the job description).\n"
            f"Candidate skills: {skills}\n"
            f"Job Title: {job_title}\n"
            f"Company: {company}\n"
            f"Job Description: {job_text}"
        )
        logger.debug("Sending prompt to OpenAI: %s", prompt[:200])
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message["content"]  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI API call failed: %s", exc)
            # Fallback to placeholder on error
            placeholder = PlaceholderProvider()
            return placeholder.judge(resume_json, job_text, job_title, company)
        # Attempt to parse JSON
        import json
        try:
            data = json.loads(content)
            score = float(data.get("score", 0.0))
            reasons = data.get("reasons", [])
            missing = data.get("missing", [])
            return score, reasons, missing
        except Exception:
            logger.warning("Failed to parse OpenAI response, using fallback")
            placeholder = PlaceholderProvider()
            return placeholder.judge(resume_json, job_text, job_title, company)


class GeminiProvider(LLMProvider):
    """Provider that uses Google Generative AI (Gemini) via google‑generativeai."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro") -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai package is required for GeminiProvider. Install it via pip."
            ) from exc
        self.genai = genai
        # API key resolution: explicit argument > env variables
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # Model resolution: environment variable GEMINI_MODEL or GOOGLE_MODEL overrides default
        env_model = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL")
        self.model_name = env_model or model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY not provided")
        # Configure client
        self.genai.configure(api_key=self.api_key)
        try:
            self.model = self.genai.GenerativeModel(self.model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Gemini model {self.model_name}: {exc}")

    def judge(self, resume_json: Dict[str, object], job_text: str, job_title: str, company: str) -> Tuple[float, List[str], List[str]]:
        skills = ", ".join(resume_json.get("skills", []))
        prompt = (
            "You are a recruiting assistant. Given a candidate résumé and a job description, "
            "assess how well the candidate fits the role. Return a JSON object with keys "
            "'score' (0 to 1), 'reasons' (list of strings explaining the match), and 'missing' "
            "(list of required skills missing from the job description).\n"
            f"Candidate skills: {skills}\n"
            f"Job Title: {job_title}\n"
            f"Company: {company}\n"
            f"Job Description: {job_text}"
        )
        logger.debug("Sending prompt to Gemini: %s", prompt[:200])
        try:
            response = self.model.generate_content(prompt)
            content = response.text  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini API call failed: %s", exc)
            placeholder = PlaceholderProvider()
            return placeholder.judge(resume_json, job_text, job_title, company)
        import json
        try:
            data = json.loads(content)
            score = float(data.get("score", 0.0))
            reasons = data.get("reasons", [])
            missing = data.get("missing", [])
            return score, reasons, missing
        except Exception:
            logger.warning("Failed to parse Gemini response, using fallback")
            placeholder = PlaceholderProvider()
            return placeholder.judge(resume_json, job_text, job_title, company)


def get_default_provider() -> LLMProvider:
    """Return an LLMProvider instance based on configuration and API keys.

    The resolution order is:

    1. If the ``LLM_PROVIDER`` environment variable is set to
       ``"openai"``, ``"gemini"`` or ``"placeholder"``, the
       corresponding provider is selected.  If the specified provider
       cannot be initialised (e.g. missing API key or package), a
       warning is logged and the automatic detection logic is used.
    2. If ``OPENAI_API_KEY`` is present, return :class:`OpenAIProvider`.
    3. If ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` is present, return
       :class:`GeminiProvider`.
    4. Otherwise, return :class:`PlaceholderProvider`.

    Returns:
        An instance of :class:`LLMProvider`.
    """
    preferred = os.getenv("LLM_PROVIDER")
    if preferred:
        pref = preferred.lower()
        if pref == "openai":
            try:
                return OpenAIProvider(os.getenv("OPENAI_API_KEY"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM_PROVIDER=openai but failed to initialise OpenAIProvider: %s", exc)
        elif pref == "gemini":
            try:
                return GeminiProvider(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM_PROVIDER=gemini but failed to initialise GeminiProvider: %s", exc)
        elif pref == "placeholder":
            logger.info("LLM_PROVIDER=placeholder; using placeholder provider")
            return PlaceholderProvider()
        else:
            logger.warning("Unknown LLM_PROVIDER value '%s'; falling back to automatic detection", preferred)
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key_openai:
        try:
            return OpenAIProvider(api_key_openai)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise OpenAIProvider: %s", exc)
    if api_key_gemini:
        try:
            return GeminiProvider(api_key_gemini)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise GeminiProvider: %s", exc)
    logger.info("No LLM API keys found; using placeholder provider")
    return PlaceholderProvider()