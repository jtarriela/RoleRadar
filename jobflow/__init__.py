"""
Jobflow package for the RoleRadar MVP CLI.

This package contains submodules for collecting job postings,
normalizing raw HTML into structured rows, parsing resumes, generating
embeddings, ranking matches, and aggregating results.  Each submodule
implements a specific step in the pipeline described in the MVP README.

The high‑level flow is:

1. **collect** – Crawl company career pages and write raw HTML into a
   local cache.  External adapters (e.g. greenhouse_adapter or
   lever_adapter) are responsible for speaking to specific job
   provider APIs when available.  The generic collector wraps a
   scraping engine like Scrythe and imposes rate limits and polite
   crawling behaviour.
2. **normalize** – Convert cached HTML pages into a normalized
   `JobRow` dataclass.  This step extracts fields such as company,
   title, location and compensation using a mixture of rules and
   optional LLM prompts for fallbacks.  The resulting rows are
   written to `jobs.csv`.
3. **resume** – Parse an unstructured résumé into a structured JSON
   representation and build an embedding vector for later ranking.
4. **rank** – Filter, vector‑rank and optionally judge each job
   against the résumé.  Weighted scores from the prefilter, vector
   similarity and LLM judge are aggregated into a final `fit_score`.
5. **cli** – Command line entry point wiring together the above
   components.

This skeleton provides simple implementations that can be extended
with real scraping logic, LLM integration and advanced ranking.  See
`README.md` for acceptance criteria and data schemas.
"""

from importlib import metadata  # noqa: F401 (expose package version)