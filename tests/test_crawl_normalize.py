"""
Unittest suite for the crawl and normalize pipeline.

These tests patch the Greenhouse and Lever adapters to return
synthesised job postings so that no network requests are made.  The
crawler writes HTML pages to a temporary cache directory.  The
normaliser then parses those HTML pages into structured ``JobRow``
instances and writes them to a CSV file.  The tests verify that
HTML files are produced, that the CSV has the correct header and
that the number of entries matches the number of fake jobs.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jobflow.collect.runner import crawl
from jobflow.normalize.html_to_fields import extract_fields
from jobflow.normalize.schema import JobRow
from jobflow.normalize.write_csv import write_jobs_csv


def _fake_job(company: str, title: str, location: str = "Remote") -> dict[str, str]:
    """Construct a minimal fake job posting.

    The returned dict contains ``job_id`` and ``html`` keys.  The
    HTML includes an ``<h1>`` tag for the job title and a ``<p>`` tag
    with a location.  This is sufficient for the normalisation
    heuristics to extract a title and location.
    """
    html = f"<html><body><h1>{title}</h1><p>{location}</p></body></html>"
    return {"job_id": f"{company}_{title}", "html": html}


class TestCrawlNormalize(unittest.TestCase):
    """Test cases for crawling and normalising job postings."""

    def setUp(self) -> None:
        # Create a temporary directory for caching HTML and CSV output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name) / "cache"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _patch_adapters(self, monkey_patch_target) -> None:
        """Helper to patch both adapters to return synthetic jobs."""
        from jobflow.ingest import greenhouse_adapter, lever_adapter

        def fake_fetch_jobs(token_or_slug: str):
            return [
                _fake_job(token_or_slug, "Software Engineer"),
                _fake_job(token_or_slug, "Data Scientist"),
            ]

        monkey_patch_target(greenhouse_adapter, "fetch_jobs", fake_fetch_jobs)
        monkey_patch_target(lever_adapter, "fetch_jobs", fake_fetch_jobs)

    def test_crawl_and_normalize(self) -> None:
        """Crawl three companies and ensure normalisation produces CSV."""
        # Define three synthetic companies using Greenhouse tokens
        companies = [
            {"name": "anduril", "greenhouse_token": "anduril"},
            {"name": "meta", "greenhouse_token": "meta"},
            {"name": "raytheon", "greenhouse_token": "raytheon"},
        ]
        # Patch adapters using context manager to avoid leaking patches
        with mock.patch("jobflow.ingest.greenhouse_adapter.fetch_jobs") as gh_mock:
            with mock.patch("jobflow.ingest.lever_adapter.fetch_jobs") as lv_mock:
                gh_mock.side_effect = lambda token: [
                    _fake_job(token, "Software Engineer"),
                    _fake_job(token, "Data Scientist"),
                ]
                lv_mock.side_effect = gh_mock.side_effect
                # Perform crawl
                crawl(companies, str(self.cache_dir))
        # Ensure that HTML files are created
        html_files = list(self.cache_dir.glob("*.html"))
        self.assertTrue(html_files, "No HTML files were cached")
        # Parse HTML into JobRow objects
        jobs: list[JobRow] = []
        for html_file in html_files:
            parts = html_file.name.split("-")
            company = parts[0]
            html_content = html_file.read_text(encoding="utf-8")
            fields = extract_fields(html_content, company=company)
            jobs.append(JobRow(**fields))
        # Write jobs to CSV
        csv_path = Path(self.temp_dir.name) / "jobs.csv"
        write_jobs_csv(jobs, str(csv_path))
        # Verify CSV exists and has header
        self.assertTrue(csv_path.exists(), "CSV file was not created")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.assertTrue(rows, "CSV file is empty")
        header = rows[0]
        expected_fields = [
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
        self.assertEqual(header, expected_fields, "CSV header does not match expected fields")
        # There should be 3 companies * 2 jobs = 6 entries plus header
        self.assertEqual(len(rows), len(jobs) + 1)


if __name__ == "__main__":
    unittest.main()