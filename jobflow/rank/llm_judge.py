"""
LLM judging stage.

This module defines a function that takes résumé and job text and
returns a relevance score along with qualitative reasons and missing
must‑have skills.  The real implementation would send a prompt to
an LLM (e.g. GPT‑4) as described in the README.  The placeholder
here simply returns a neutral score and boilerplate reasons.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

from ..normalize.schema import JobRow
from .llm_providers import get_default_provider, LLMProvider, PlaceholderProvider

logger = logging.getLogger(__name__)


def judge_jobs(jobs: Iterable[JobRow], resume_json: dict) -> List[Tuple[JobRow, float, List[str], List[str]]]:
    """Judge the relevance of jobs using a configured LLM provider.

    This function delegates to an LLM provider (OpenAI, Gemini or a
    placeholder) to assess how well a résumé matches each job.  The
    provider is selected based on available API keys via
    :func:`get_default_provider`.

    Args:
        jobs: Iterable of JobRow objects.
        resume_json: Parsed résumé JSON dictionary.

    Returns:
        A list of tuples `(job, relevance, reasons, must_have_gaps)`.
        Relevance is a float between 0 and 1.  Reasons is a list of
        strings explaining the match.  `must_have_gaps` lists the
        skills missing from the job that the résumé specifies as
        required.
    """
    provider: LLMProvider = get_default_provider()
    results: List[Tuple[JobRow, float, List[str], List[str]]] = []
    for job in jobs:
        try:
            score, reasons, missing = provider.judge(
                resume_json,
                job.description_text,
                job.title,
                job.company,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM provider failed for job %s: %s", job.job_id, exc)
            placeholder = PlaceholderProvider()
            score, reasons, missing = placeholder.judge(
                resume_json,
                job.description_text,
                job.title,
                job.company,
            )
        results.append((job, score, reasons, missing))
    logger.debug("Judged %d jobs via LLM provider %s", len(results), provider.__class__.__name__)
    return results