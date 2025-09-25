"""
Ranking subsystem for RoleRadar.

The `rank` package combines multiple scoring stages to produce a
final fit score for each job.  The stages include:

* `prefilter` – Filters out jobs that obviously do not meet the
  candidate's preferences (e.g. wrong location or compensation).
* `vector_rank` – Computes cosine similarity between embeddings of
  the résumé and each job description.
* `llm_judge` – Optionally calls an LLM to judge the relevance and
  provide qualitative reasons.
* `aggregate` – Combines the scores from the above stages using
  configured weights.
"""

from .prefilter import prefilter_jobs  # noqa: F401
from .vector_rank import build_job_embeddings, rank_by_vector  # noqa: F401
from .llm_judge import judge_jobs  # noqa: F401
from .aggregate import aggregate_scores  # noqa: F401