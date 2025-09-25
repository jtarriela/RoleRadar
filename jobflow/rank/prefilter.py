"""
Prefiltering stage for ranking.

This module filters out jobs that do not meet the candidate's
preferences before performing expensive scoring.  Preferences may
include desired locations, minimum compensation and remote
acceptability.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from ..normalize.schema import JobRow

logger = logging.getLogger(__name__)


def prefilter_jobs(jobs: Iterable[JobRow], preferences: Dict[str, object]) -> List[JobRow]:
    """Return jobs that satisfy the candidate's basic preferences.

    Args:
        jobs: Iterable of JobRow objects.
        preferences: Dictionary that may contain keys:
            `locations` (list of acceptable location strings),
            `min_pay` (float) and `remote_ok` (bool).

    Returns:
        A list of jobs that pass the filter.
    """
    allowed_locs = [loc.lower() for loc in preferences.get("locations", [])]
    min_pay = preferences.get("min_pay")
    remote_ok = preferences.get("remote_ok", False)
    filtered: List[JobRow] = []
    for job in jobs:
        # Check pay
        if min_pay is not None:
            if job.pay_max is not None and job.pay_max < min_pay:
                logger.debug("Filtering out %s due to pay", job.job_id)
                continue
        # Check location
        if allowed_locs:
            job_locs = [loc.lower() for loc in job.locations]
            if not any(
                (loc in job_locs) or (loc == "remote" and job.remote_flag)
                for loc in allowed_locs
            ):
                logger.debug("Filtering out %s due to location", job.job_id)
                continue
        # Check remote flag
        if not remote_ok and job.remote_flag:
            logger.debug("Filtering out %s due to remote flag", job.job_id)
            continue
        filtered.append(job)
    logger.info("Prefiltered %d -> %d jobs", len(list(jobs)), len(filtered))
    return filtered