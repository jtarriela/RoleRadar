# ingest/greenhouse_adapter.py
from __future__ import annotations
import hashlib, datetime, requests
from typing import List
from bs4 import BeautifulSoup
from normalize.schema import JobRow

def strip_html(html: str) -> str:
    text = BeautifulSoup(html or "", "lxml").get_text(" ")
    return " ".join(text.split())

def sha(url: str, text_head: str) -> str:
    return hashlib.sha256((url + "|" + text_head).encode("utf-8")).hexdigest()

def fetch(company_token: str, timeout: int = 30) -> List[JobRow]:
    """
    company_token is the 'board token' (e.g., 'openai').
    """
    url = f"https://boards-api.greenhouse.io/v1/boards/{company_token}/jobs?content=true"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    jobs = data.get("jobs", [])
    now_iso = datetime.datetime.utcnow().isoformat()

    rows: List[JobRow] = []
    for j in jobs:
        desc_html = j.get("content") or ""
        desc_text = strip_html(desc_html)
        absolute_url = j.get("absolute_url") or ""
        locations = []
        if isinstance(j.get("location"), dict):
            name = j["location"].get("name") or ""
            if name:
                locations = [name]

        fp = sha(absolute_url, desc_text[:2048])

        rows.append(JobRow(
            job_id=None,
            company=company_token,
            source="greenhouse",
            ext_id=str(j.get("id")) if j.get("id") is not None else None,
            title=j.get("title") or "",
            department=None,                       # can be derived from offices/departments endpoints later
            employment_type=None,                  # GH doesn't always provide
            locations=locations,
            remote_flag=("remote" in " ".join(locations).lower()) if locations else False,
            onsite_flag=not (("remote" in " ".join(locations).lower()) if locations else False),
            pay_min=None, pay_max=None, pay_currency=None,
            posted_date=(j.get("updated_at") or "")[:10] or None,
            updated_at=j.get("updated_at"),
            absolute_url=absolute_url,
            description_text=desc_text,
            description_html=desc_html,
            scraped_at=now_iso,
            source_fingerprint=fp
        ))
    return rows
