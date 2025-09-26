"""
Command line interface for RoleRadar.

This module exposes subcommands to run each stage of the pipeline: parsing a
résumé, crawling careers pages, normalizing HTML into CSV, matching a
candidate résumé against job postings and generating a human‑readable
report.  The CLI is intentionally lightweight and delegates most of the
work to functions in the `collect`, `normalize`, `resume` and `rank`
packages.

Compared to the previous version, the résumé parse command no longer
supports MinerU preprocessing; instead the default behaviour is to rely on
an LLM provider (preferably Gemini 1.5 Pro) for parsing, with a
conservative heuristic fallback when no provider is configured.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import yaml  # type: ignore

from .collect.runner import crawl  # type: ignore
from .normalize.html_to_fields import extract_fields  # type: ignore
from .normalize.schema import JobRow  # type: ignore
from .normalize.write_csv import write_jobs_csv  # type: ignore
from .resume.parse_resume import parse_resume, save_resume_json  # type: ignore
from .resume.embed import embed_resume  # type: ignore
from .rank.prefilter import prefilter_jobs  # type: ignore
from .rank.vector_rank import rank_by_vector  # type: ignore
from .rank.llm_judge import judge_jobs  # type: ignore
from .rank.aggregate import aggregate_scores  # type: ignore

logger = logging.getLogger("jobflow.cli")


def _load_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_resume_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jobs_csv(path: str) -> List[JobRow]:
    jobs: List[JobRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert CSV strings back into appropriate types
            locations = row.get("locations", "").split(";") if row.get("locations") else []
            jobs.append(
                JobRow(
                    job_id=row.get("job_id", ""),
                    company=row.get("company", ""),
                    source=row.get("source", ""),
                    ext_id=row.get("ext_id", ""),
                    title=row.get("title", ""),
                    department=row.get("department") or None,
                    employment_type=row.get("employment_type") or None,
                    locations=locations,
                    remote_flag=row.get("remote_flag", "False") == "True",
                    onsite_flag=row.get("onsite_flag", "False") == "True",
                    pay_min=float(row["pay_min"]) if row.get("pay_min") else None,
                    pay_max=float(row["pay_max"]) if row.get("pay_max") else None,
                    pay_currency=row.get("pay_currency") or None,
                    posted_date=row.get("posted_date") or None,
                    updated_at=row.get("updated_at") or None,
                    absolute_url=row.get("absolute_url", ""),
                    description_text=row.get("description_text", ""),
                    description_html=row.get("description_html", ""),
                    scraped_at=row.get("scraped_at", ""),
                    source_fingerprint=row.get("source_fingerprint", ""),
                )
            )
    return jobs


def cmd_resume_parse(args: argparse.Namespace) -> None:
    """Parse a résumé file and write JSON plus embedding.

    This command attempts to parse the résumé using an LLM provider by
    default.  If ``--no-use-llm`` is provided or the LLM provider is not
    configured, a simple heuristic parser is used.  After parsing, the
    résumé's skills are embedded into a vector representation and the
    complete JSON is written to the specified output path.
    """
    resume = parse_resume(args.file, use_llm=args.use_llm)
    # Compute embedding on skills (simple vector) and attach.
    resume.embedding = embed_resume(resume.skills)
    save_resume_json(resume, args.out)
    logger.info("Resume parsed and saved to %s", args.out)


def cmd_crawl(args: argparse.Namespace) -> None:
    """Run the crawl stage using a YAML config."""
    cfg = _load_config(args.config)
    companies = cfg.get("companies", [])
    crawl(companies, args.cache)
    logger.info("Crawl complete; cache written to %s", args.cache)


def cmd_normalize(args: argparse.Namespace) -> None:
    """Normalize cached HTML into CSV."""
    cache_dir = Path(args.cache)
    html_files = list(cache_dir.glob("*.html"))
    jobs: List[JobRow] = []
    for html_file in html_files:
        with open(html_file, "r", encoding="utf-8") as f:
            html = f.read()
        # Attempt to derive company from filename
        parts = html_file.stem.split("-")
        company = parts[0] if parts else ""
        fields = extract_fields(html, company=company, url="")
        jobs.append(JobRow(**fields))
    write_jobs_csv(jobs, args.out)
    logger.info("Normalized %d pages into %s", len(jobs), args.out)


def cmd_match(args: argparse.Namespace) -> None:
    """Match résumé to jobs and write matches CSV."""
    resume_json = _load_resume_json(args.resume)
    jobs = _load_jobs_csv(args.jobs)
    # Build preferences dictionary
    preferences: Dict[str, object] = {}
    if args.location:
        preferences["locations"] = [loc.strip() for loc in args.location.split(",")]
    if args.min_pay is not None:
        preferences["min_pay"] = args.min_pay
    preferences["remote_ok"] = args.remote_ok
    # Prefilter jobs
    filtered_jobs = prefilter_jobs(jobs, preferences)
    if not filtered_jobs:
        logger.warning("No jobs matched basic preferences")
    # Compute vector scores
    resume_embedding = resume_json.get("embedding") or embed_resume(resume_json.get("skills", []))
    vector_scores = rank_by_vector(filtered_jobs, resume_embedding)
    # Judge via LLM (placeholder fallback)
    llm_scores = judge_jobs(filtered_jobs, resume_json)
    # Aggregate scores using weights and threshold
    weights = {
        "prefilter": 0.30,
        "vector": 0.45,
        "llm": 0.25,
    }
    aggregated = aggregate_scores(filtered_jobs, vector_scores, llm_scores, weights)
    # Apply threshold and top‑k
    threshold = args.threshold or 0.0
    topk = args.topk or len(aggregated)
    selected = [row for row in aggregated if row["fit_score"] >= threshold][:topk]
    # Write matches CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "job_id",
            "company",
            "title",
            "fit_score",
            "reasons",
            "must_have_gaps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in selected:
            job: JobRow = entry["job"]
            writer.writerow(
                {
                    "job_id": job.job_id,
                    "company": job.company,
                    "title": job.title,
                    "fit_score": f"{entry['fit_score']:.4f}",
                    "reasons": "; ".join(entry["reasons"]),
                    "must_have_gaps": "; ".join(entry["must_have_gaps"]),
                }
            )
    logger.info("Wrote %d matches to %s", len(selected), args.out)


def cmd_report(args: argparse.Namespace) -> None:
    """Print a simple report from matches CSV."""
    with open(args.matches, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    limit = args.limit or len(rows)
    for i, row in enumerate(rows[:limit]):
        fit_score = float(row["fit_score"])
        print(f"{i+1:02d}. {row['title']} at {row['company']} – {fit_score:.2%}")
        print(f"   Reasons: {row['reasons']}")
        if row['must_have_gaps']:
            print(f"   Gaps: {row['must_have_gaps']}")
        print()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jobflow", description="RoleRadar CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Resume parse
    resume_parser = subparsers.add_parser("resume", help="Résumé related commands")
    resume_sub = resume_parser.add_subparsers(dest="subcommand", required=True)
    parse_resume_cmd = resume_sub.add_parser("parse", help="Parse a résumé file")
    parse_resume_cmd.add_argument("--file", required=True, help="Path to résumé file (txt, pdf, doc, docx)")
    parse_resume_cmd.add_argument("--out", required=True, help="Path to output JSON file")
    # LLM parsing flags: --use-llm (default) and --no-use-llm to disable
    llm_group = parse_resume_cmd.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--use-llm",
        dest="use_llm",
        action="store_true",
        default=True,
        help="Enable LLM parsing for résumés (default)",
    )
    llm_group.add_argument(
        "--no-use-llm",
        dest="use_llm",
        action="store_false",
        help="Disable LLM parsing and use the heuristic parser",
    )
    parse_resume_cmd.set_defaults(func=cmd_resume_parse)

    # Crawl
    crawl_cmd = subparsers.add_parser("crawl", help="Crawl careers pages")
    crawl_cmd.add_argument("--config", default="jobflow/config.yaml", help="YAML config file")
    crawl_cmd.add_argument("--cache", default=".cache", help="Cache directory for raw HTML")
    crawl_cmd.set_defaults(func=cmd_crawl)

    # Normalize
    norm_cmd = subparsers.add_parser("normalize", help="Normalize cached HTML to CSV")
    norm_cmd.add_argument("--cache", default=".cache", help="Cache directory containing HTML")
    norm_cmd.add_argument("--out", default="jobs.csv", help="Output CSV path")
    norm_cmd.set_defaults(func=cmd_normalize)

    # Match
    match_cmd = subparsers.add_parser("match", help="Match résumé to jobs")
    match_cmd.add_argument("--resume", required=True, help="Path to résumé JSON")
    match_cmd.add_argument("--jobs", required=True, help="Path to jobs CSV")
    match_cmd.add_argument("--location", help="Comma separated preferred locations")
    match_cmd.add_argument("--min-pay", type=float, dest="min_pay", help="Minimum pay")
    match_cmd.add_argument("--remote-ok", action="store_true", help="Allow remote jobs")
    match_cmd.add_argument("--threshold", type=float, default=0.0, help="Minimum fit score")
    match_cmd.add_argument("--topk", type=int, default=50, help="Maximum number of matches to output")
    match_cmd.add_argument("--out", default="matches.csv", help="Output CSV path")
    match_cmd.set_defaults(func=cmd_match)

    # Report
    report_cmd = subparsers.add_parser("report", help="Generate a text report from matches CSV")
    report_cmd.add_argument("--matches", required=True, help="Path to matches CSV")
    report_cmd.add_argument("--limit", type=int, default=20, help="Number of top matches to display")
    report_cmd.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])