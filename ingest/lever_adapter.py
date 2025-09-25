# ingest/lever_adapter.py
from __future__ import annotations
import hashlib, datetime, requests
from typing import List, Optional
from normalize.schema import JobRow
from bs4 import BeautifulSoup

def strip_html(html: str) -> str:
    text = BeautifulSoup(html or "", "lxml").get_text(" ")
    return " ".join(text.split())

def sha(url: str, text_head: str) -> str:
    return hashlib.sha256((url + "|" + text_head).encode("utf-8")).hexdigest()

def iso_from_ms(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()

def fetch(company_slug: str, timeout: int = 30) -> List[JobRow]:
    """
    company_slug is the Lever company handle (e.g., 'databricks').
    """
    url = f"https://api.lever.co/v0/postings/{company_slug}?mode=json"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    now_iso = datetime.datetime.utcnow().isoformat()

    rows: List[JobRow] = []
    for j in data:
        desc_html = j.get("description") or ""
        desc_text = strip_html(desc_html)
        cats = j.get("categories") or {}
        location = cats.get("location")
        locations = [location] if location else []
        hosted = j.get("hostedUrl") or ""

        fp = sha(hosted, desc_text[:2048])

        rows.append(JobRow(
            job_id=None,
            company=company_slug,
            source="lever",
            ext_id=j.get("id"),
            title=j.get("text") or "",
            department=cats.get("team") or cats.get("department"),
            employment_type=cats.get("commitment"),
            locations=locations,
            remote_flag=("remote" in " ".join(locations).lower()) if locations else False,
            onsite_flag=not (("remote" in " ".join(locations).lower()) if locations else False),
            pay_min=None, pay_max=None, pay_currency=None,   # Lever pay often not public; parse later from desc
            posted_date=(iso_from_ms(j.get("createdAt")) or "")[:10] or None,
            updated_at=iso_from_ms(j.get("updatedAt")),
            absolute_url=hosted,
            description_text=desc_text,
            description_html=desc_html,
            scraped_at=now_iso,
            source_fingerprint=fp
        ))
    return rows
