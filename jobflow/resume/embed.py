"""
Résumé embedding generator.

Embeddings are used to compute vector similarity between a candidate
and job descriptions.  In a production system this module would call
a pre‑trained embedding model (e.g. sentence‑transformers) or an
external API (e.g. OpenAI embedding API).  Here we implement a
simple fallback that hashes the input skills into a fixed‑length
vector.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable, List

logger = logging.getLogger(__name__)


def _hash_to_float(value: str) -> float:
    """Deterministically hash a string into a float between 0 and 1."""
    h = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def embed_resume(skills: Iterable[str], dim: int = 32) -> List[float]:
    """Generate a deterministic embedding vector from a list of skills.

    Args:
        skills: Iterable of skill names.
        dim: Length of the output vector.

    Returns:
        A list of floats representing the embedding.  Each skill
        contributes to several positions in the vector based on its
        hash.  The resulting vector is normalised to unit length.
    """
    vector = [0.0] * dim
    for skill in skills:
        base = _hash_to_float(skill)
        for i in range(dim):
            # Rotate hash for each dimension
            rotated = (base * (i + 1)) % 1.0
            vector[i] += rotated
    # Normalise vector
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    embedding = [v / norm for v in vector]
    logger.debug("Generated embedding for %s: %s", skills, embedding)
    return embedding