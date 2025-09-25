"""
Greenhouse adapter.

This module provides a function to fetch job listings from the
Greenhouse recruitment platform.  Greenhouse exposes a JSON API for
public job boards (see https://developers.greenhouse.io/).  A
`greenhouse_token` corresponds to the subdomain of a company's
Greenhouse URL (e.g. for `https://boards.greenhouse.io/openai` the
token is `openai`).  The function returns a list of dicts with at
least a `job_id` and raw `html` for each posting.

The current implementation is a placeholder that returns an empty
list.  Replace the body with a proper HTTP request (e.g. using
`requests.get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs")`) and
HTML retrieval as needed.
"""

from __future__ import annotations

from typing import List, Dict


def fetch_jobs(token: str) -> List[Dict[str, str]]:
    """Fetch job postings for a Greenhouse organization.

    Args:
        token: The Greenhouse board token (e.g. "openai").

    Returns:
        A list of job postings, each represented as a dict with
        `job_id` and `html` keys.  Additional metadata may be
        included.  In this placeholder implementation, an empty list
        is returned to indicate no jobs were fetched.
    """
    # TODO: Implement actual API call to Greenhouse boards API.
    # Example:
    #   resp = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs")
    #   for job in resp.json().get("jobs", []):
    #       job_id = str(job.get("id"))
    #       html = requests.get(job["url"]).text
    #       yield {"job_id": job_id, "html": html}
    return []