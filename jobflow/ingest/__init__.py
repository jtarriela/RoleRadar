"""
Provider adapters for RoleRadar.

This package contains thin wrappers around external job provider APIs.
Adapters accept provider‑specific identifiers (e.g. a Greenhouse
organization token or a Lever slug) and return lists of dictionaries
containing at minimum a `job_id` and raw `html`.  The exact
implementation of each adapter may vary and may use requests or
selenium.  For the MVP these functions return empty lists or mock
data; they should be replaced with real API calls or scraping
implementations.
"""

from .greenhouse_adapter import fetch_jobs as greenhouse_fetch_jobs  # noqa: F401
from .lever_adapter import fetch_jobs as lever_fetch_jobs  # noqa: F401