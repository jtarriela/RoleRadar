# normalize/schema.py
from dataclasses import dataclass, asdict
from typing import List, Optional

JOB_HEADERS = [
    "job_id", "company", "source", "ext_id", "title", "department",
    "employment_type", "locations", "remote_flag", "onsite_flag",
    "pay_min", "pay_max", "pay_currency", "posted_date", "updated_at",
    "absolute_url", "description_text", "description_html", "scraped_at",
    "source_fingerprint"
]

@dataclass
class JobRow:
    job_id: Optional[int]
    company: str
    source: str                   # 'greenhouse' | 'lever' | ...
    ext_id: Optional[str]
    title: str
    department: Optional[str]
    employment_type: Optional[str]
    locations: List[str]
    remote_flag: bool
    onsite_flag: bool
    pay_min: Optional[float]
    pay_max: Optional[float]
    pay_currency: Optional[str]
    posted_date: Optional[str]    # YYYY-MM-DD
    updated_at: Optional[str]     # ISO8601
    absolute_url: str
    description_text: str
    description_html: str
    scraped_at: str               # ISO8601
    source_fingerprint: str       # sha256(url + first_2kb_text)

    def to_csv_row(self) -> list:
        d = asdict(self)
        d["locations"] = " | ".join(self.locations) if self.locations else ""
        return [d[h] for h in JOB_HEADERS]
