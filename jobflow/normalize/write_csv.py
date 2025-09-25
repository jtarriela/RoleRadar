"""
CSV writer for normalized jobs.

Provides a helper to write a list of `JobRow` instances to a CSV
file using the field order defined by the dataclass.  If the file
already exists, it will be overwritten.  Unicode is written in
UTF‑8 encoding.
"""

from __future__ import annotations

import csv
from typing import Iterable, List

from .schema import JobRow


def write_jobs_csv(jobs: Iterable[JobRow], path: str) -> None:
    """Write normalized jobs to a CSV file.

    Args:
        jobs: Iterable of `JobRow` objects.
        path: Destination path for the CSV.
    """
    fieldnames = [
        "job_id",
        "company",
        "source",
        "ext_id",
        "title",
        "department",
        "employment_type",
        "locations",
        "remote_flag",
        "onsite_flag",
        "pay_min",
        "pay_max",
        "pay_currency",
        "posted_date",
        "updated_at",
        "absolute_url",
        "description_text",
        "description_html",
        "scraped_at",
        "source_fingerprint",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            # Convert dataclass to dict and ensure lists are JSON‑friendly
            row = job.__dict__.copy()
            row["locations"] = ";".join(job.locations)
            writer.writerow(row)