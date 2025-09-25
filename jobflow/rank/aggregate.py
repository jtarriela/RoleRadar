"""
Score aggregation for ranking.

Combines the outputs of the prefilter, vector similarity and LLM
judging stages into a final `fit_score` for each job.  The weights
for each component are configurable via a dictionary with keys
`prefilter`, `vector` and `llm` whose values sum to 1.0.  Jobs not
present in a particular component are treated as having a score of
zero for that component.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from ..normalize.schema import JobRow

logger = logging.getLogger(__name__)


def aggregate_scores(
    prefiltered: Iterable[JobRow],
    vector_scores: Iterable[Tuple[JobRow, float]],
    llm_scores: Iterable[Tuple[JobRow, float, List[str], List[str]]],
    weights: Dict[str, float],
) -> List[Dict[str, object]]:
    """Aggregate scores from prefilter, vector and LLM stages.

    Args:
        prefiltered: Jobs that passed the prefilter stage.
        vector_scores: Tuples of (job, cosine_similarity).
        llm_scores: Tuples of (job, relevance, reasons, gaps).
        weights: Dict with weight values for 'prefilter', 'vector' and 'llm'.

    Returns:
        A list of dictionaries with keys: job, fit_score, reasons.
    """
    weight_pref = weights.get("prefilter", 0.0)
    weight_vec = weights.get("vector", 0.0)
    weight_llm = weights.get("llm", 0.0)
    # Build lookup for vector and llm scores.
    vector_lookup: Dict[str, float] = {job.job_id: score for job, score in vector_scores}
    llm_lookup: Dict[str, Tuple[float, List[str], List[str]]] = {
        job.job_id: (rel, reasons, gaps) for job, rel, reasons, gaps in llm_scores
    }
    aggregated: List[Dict[str, object]] = []
    for job in prefiltered:
        vec_score = vector_lookup.get(job.job_id, 0.0)
        llm_entry = llm_lookup.get(job.job_id, (0.0, [], []))
        llm_score, reasons, gaps = llm_entry
        # Prefilter contributes 1.0 if job passed, 0.0 otherwise.
        pref_score = 1.0
        fit_score = (
            weight_pref * pref_score
            + weight_vec * vec_score
            + weight_llm * llm_score
        )
        aggregated.append(
            {
                "job": job,
                "fit_score": fit_score,
                "reasons": reasons,
                "must_have_gaps": gaps,
            }
        )
    # Sort by fit_score descending
    aggregated.sort(key=lambda x: x["fit_score"], reverse=True)
    logger.debug("Aggregated scores for %d jobs", len(aggregated))
    return aggregated