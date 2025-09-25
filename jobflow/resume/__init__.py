"""
Resume parsing and embedding utilities.

This package contains functions to convert an unstructured résumé
into a structured JSON document and to compute an embedding vector
for ranking.  The parsing step uses pattern matching and may call
LLM providers when configured.  Embeddings can be generated from a
pre‑trained model or via an external API.
"""

from .parse_resume import parse_resume, save_resume_json  # noqa: F401
from .embed import embed_resume  # noqa: F401