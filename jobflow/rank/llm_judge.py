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

logger = logging.getLogger(__name__)


def judge_jobs(jobs: Iterable[JobRow], resume_json: dict) -> List[Tuple[JobRow, float, List[str], List[str]]]:
    """Judge the relevance of jobs using an LLM (placeholder).

    Args:
        jobs: Iterable of JobRow objects.
        resume_json: Parsed résumé JSON dictionary.

    Returns:
        A list of tuples `(job, relevance, reasons, must_have_gaps)`.
        Relevance is a float between 0 and 1.  Reasons is a list of
        strings explaining the match.  `must_have_gaps` lists the
        skills missing from the job that the résumé specifies as
        required.  In this placeholder implementation, relevance is
        fixed at 0.5 and the reasons are generic.
    """
    results: List[Tuple[JobRow, float, List[str], List[str]]] = []
    skills = set(resume_json.get("skills", []))
    for job in jobs:
        # Determine missing must‑have skills: any resume skill not in job description.
        missing = [s for s in skills if s.lower() not in job.description_text.lower()]
        reasons = [
            f"Basic role alignment with {job.title}",
            f"Company {job.company} and candidate skills overlap", 
        ]
        relevance = 0.5  # neutral placeholder
        results.append((job, relevance, reasons, missing))
    logger.debug("Judged %d jobs via LLM placeholder", len(results))
    return results