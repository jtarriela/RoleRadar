"""
HTML to fields extractor.

This module provides helper functions to parse raw HTML from a job
posting into a dictionary of fields defined by `JobRow`.  A simple
rule‑based extractor using BeautifulSoup is provided.  When the
heuristics fail to locate a field, the value is set to `None` or an
empty string.  A future version may incorporate LLM calls to handle
arbitrary page structures as a fallback.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


def extract_fields(html: str, *, company: Optional[str] = None, url: str | None = None) -> Dict[str, object]:
    """Extract key fields from a job posting's HTML.

    Args:
        html: Raw HTML of the job posting.
        company: Optional company name hint.
        url: Optional canonical URL of the posting.

    Returns:
        A dictionary with keys matching `JobRow` fields.  Unknown
        values are set to `None` or an empty string.  The
        `source_fingerprint` field is computed as an md5 hash of the
        HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Title: look for <h1> or <h2> tags with job titles.
    title_tag = soup.find(["h1", "h2"])
    title = title_tag.get_text(strip=True) if title_tag else ""
    # Plain text description: join all paragraph text.
    paragraphs: List[str] = [p.get_text(strip=True) for p in soup.find_all("p")]
    description_text = "\n".join(paragraphs)
    description_html = str(soup)
    # Locations: naive extraction for words containing known city/remote keywords.
    location_candidates: List[str] = []
    for loc in soup.find_all(text=re.compile(r"(remote|san|new york|london|paris)", re.I)):
        location_candidates.append(loc.strip())
    locations = list({lc for lc in location_candidates if lc})
    remote_flag = any(re.search(r"remote", loc, re.I) for loc in locations)
    onsite_flag = not remote_flag
    # Pay range: look for patterns like "$100k" or "120,000".
    pays = re.findall(r"\$?([0-9][0-9,]+)\s*(k|K)?", html)
    pay_vals = []
    for val, suffix in pays:
        num = float(val.replace(",", ""))
        if suffix.lower() == "k":
            num *= 1_000
        pay_vals.append(num)
    pay_min = min(pay_vals) if pay_vals else None
    pay_max = max(pay_vals) if pay_vals else None
    # Generate fingerprint of HTML.
    fingerprint = hashlib.md5(html.encode("utf-8")).hexdigest()
    # Timestamp for scraped_at.
    scraped_at = datetime.utcnow().isoformat()
    return {
        "job_id": fingerprint[:12],
        "company": company or "",
        "source": "html",
        "ext_id": fingerprint[:12],
        "title": title,
        "department": None,
        "employment_type": None,
        "locations": locations,
        "remote_flag": remote_flag,
        "onsite_flag": onsite_flag,
        "pay_min": pay_min,
        "pay_max": pay_max,
        "pay_currency": "USD" if pay_vals else None,
        "posted_date": None,
        "updated_at": None,
        "absolute_url": url or "",
        "description_text": description_text,
        "description_html": description_html,
        "scraped_at": scraped_at,
        "source_fingerprint": fingerprint,
    }