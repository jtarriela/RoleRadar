"""
Job collection runner.

This module exposes a `crawl` function that orchestrates fetching
job listings from multiple companies.  It uses provider‑specific
adapters (e.g. Greenhouse or Lever) when hints are provided in the
configuration.  For custom or unknown portals it can call a
user‑supplied scraping engine.  Collected raw HTML is written to a
cache directory which the normalize step reads from.

Note: This implementation is intentionally simple and does not
include full crawling logic.  It serves as a placeholder for the
MVP and demonstrates how to structure the collector.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional

# Provider adapters live in the ingest package.
from ..ingest import greenhouse_adapter, lever_adapter

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _write_cache(company: str, job_id: str, html: str, cache_dir: str) -> None:
    """Write a single HTML page to the cache.

    Cached files are named `{company}-{job_id}.html` and placed in
    `cache_dir`.  A timestamp is prepended to avoid name collisions.
    """
    _ensure_dir(cache_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{company}-{job_id}-{timestamp}.html"
    path = os.path.join(cache_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.debug("Cached %s to %s", job_id, path)


def crawl(companies: List[Dict[str, str]], cache_dir: str, *,
          delay_ms_range: Optional[Iterable[int]] = None) -> None:
    """Crawl the given companies' career sites and store raw HTML.

    Args:
        companies: A list of dicts describing each company.  Each
            entry may include a `name`, and hints like
            `greenhouse_token`, `lever_slug` or a generic
            `careers_url`.  See `config.yaml` for examples.
        cache_dir: Directory where raw HTML files will be stored.
        delay_ms_range: Optional two‑element iterable `[min_ms, max_ms]`
            specifying random dwell times between requests.

    This function iterates through the companies and dispatches to
    provider‑specific adapters when hints are available.  If no hint
    matches, it logs that the provider is unsupported.  Actual
    crawling logic should be implemented in the adapters.
    """
    logger.info("Starting crawl for %d companies", len(companies))
    stats = {
        "start_time": datetime.utcnow().isoformat(),
        "companies": len(companies),
        "pages_fetched": 0,
        "errors": 0,
    }
    for company in companies:
        name = company.get("name", "unknown")
        try:
            if "greenhouse_token" in company:
                logger.debug("Using Greenhouse adapter for %s", name)
                pages = greenhouse_adapter.fetch_jobs(company["greenhouse_token"])
            elif "lever_slug" in company:
                logger.debug("Using Lever adapter for %s", name)
                pages = lever_adapter.fetch_jobs(company["lever_slug"])
            elif "careers_url" in company:
                logger.warning(
                    "No adapter implemented for careers_url; skipping %s", name
                )
                pages = []
            else:
                logger.warning("Unsupported company configuration: %s", company)
                pages = []
            for page in pages:
                job_id = page.get("job_id") or page.get("id") or "unknown"
                html = page.get("html") or ""
                _write_cache(name, job_id, html, cache_dir)
                stats["pages_fetched"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error crawling %s: %s", name, exc)
            stats["errors"] += 1
    stats["end_time"] = datetime.utcnow().isoformat()
    # Write a simple log for the crawl.
    log_path = os.path.join(cache_dir, "collect.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats) + "\n")
    logger.info("Crawl finished: %s", stats)