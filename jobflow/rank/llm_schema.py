"""
LLM judgement schema.

Defines a dataclass to represent the outcome of evaluating a résumé
against a job description using a large language model.  Each
`LLMJudgement` contains a relevance score between 0 and 1, a list
of reasons supporting the score, and a list of skills missing from
the job description relative to the candidate's must‑have skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class LLMJudgement:
    """Result of an LLM job match evaluation."""

    score: float
    reasons: List[str]
    missing: List[str]