"""
Normalization subsystem for RoleRadar.

This package converts raw HTML pages into structured `JobRow`
instances and writes them to CSV files.  It also provides helper
functions to extract fields from HTML using rule‑based parsing and
optional LLM fallbacks when the heuristics fail.

The normalized CSV format is defined by the `JobRow` dataclass in
`schema.py`.  See the README for the list of required headers.
"""

from .schema import JobRow  # noqa: F401
from .html_to_fields import extract_fields  # noqa: F401
from .write_csv import write_jobs_csv  # noqa: F401