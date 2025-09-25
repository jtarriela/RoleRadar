"""
Vector ranking stage.

This module provides utilities to compute similarity scores between a
rédumé embedding and job embeddings.  A simple term hashing
approach is used for the MVP.  In a production system you could
replace this with sentence embeddings (e.g. via sentence‑transformers)
or a vector database like pgvector.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Iterable, List, Tuple

from ..normalize.schema import JobRow

logger = logging.getLogger(__name__)


def _embed_text(text: str, dim: int = 32) -> List[float]:
    """Embed text into a fixed size vector using hashing.

    Splits text on whitespace and hashes each token to contribute to
    the embedding.  This is a deterministic and inexpensive method
    suitable for the MVP.
    """
    tokens = text.lower().split()
    vector = [0.0] * dim
    for token in tokens:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        base = int(h[:8], 16) / 0xFFFFFFFF
        for i in range(dim):
            rotated = (base * (i + 1)) % 1.0
            vector[i] += rotated
    # Normalise
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


def build_job_embeddings(jobs: Iterable[JobRow], dim: int = 32) -> Dict[str, List[float]]:
    """Build an embedding for each job's description.

    Args:
        jobs: Iterable of JobRow objects.
        dim: Dimension of the embeddings.

    Returns:
        A mapping from job_id to embedding vector.
    """
    embeddings: Dict[str, List[float]] = {}
    for job in jobs:
        embeddings[job.job_id] = _embed_text(job.description_text, dim)
    logger.debug("Built embeddings for %d jobs", len(embeddings))
    return embeddings


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rank_by_vector(jobs: Iterable[JobRow], resume_embedding: List[float], dim: int = 32) -> List[Tuple[JobRow, float]]:
    """Rank jobs by cosine similarity to the résumé embedding.

    Args:
        jobs: Iterable of JobRow objects.
        resume_embedding: Embedding vector for the résumé.
        dim: Dimensionality used to build job embeddings (should match
            the resume embedding length).

    Returns:
        A list of tuples `(job, score)` sorted in descending order of
        similarity.
    """
    job_embeddings = build_job_embeddings(jobs, dim)
    ranked: List[Tuple[JobRow, float]] = []
    for job in jobs:
        job_emb = job_embeddings.get(job.job_id)
        if job_emb:
            score = _cosine_similarity(resume_embedding, job_emb)
            ranked.append((job, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    logger.debug("Ranked %d jobs by vector similarity", len(ranked))
    return ranked