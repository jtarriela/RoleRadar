"""
Collection subsystem for RoleRadar.

The `collect` package wraps the scraping layer.  It provides a simple
`crawl` function that dispatches to provider‑specific adapters when
available (e.g. Greenhouse, Lever) or falls back to a generic HTML
scraper like Scrythe.  Each run stores raw pages in a local cache
directory and records basic session metrics in a log file.

The collector respects the anti‑bot posture defined in the MVP
specification by limiting concurrency, randomising delays and
maintaining an idempotent cache.  If a page has already been
downloaded today, it will not be fetched again.
"""

from .runner import crawl  # noqa: F401