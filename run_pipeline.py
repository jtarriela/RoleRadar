#!/usr/bin/env python3
"""Unified RoleRadar pipeline driven by a YAML configuration file.

This script orchestrates the full data workflow:

1. Optional sitemap extraction for each target site
2. Markdown scraping with crawl4ai-based scrapers
3. Async LLM parsing of job markdown into structured JSON
4. Résumé parsing (LLM-assisted with fallback heuristics)
5. Structured résumé ↔ job matching with detailed breakdowns

All intermediate artefacts plus final matches are written to the run-specific
output directory defined in the YAML configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

from src.jobdata_markdown_json_concurrent import AsyncJobProcessor, ProcessingStats
from src.raw_job_scraper import BaseJobScraper, ScrapingConfig, scrape_company
from src.resume_parser import parse_resume, save_resume_json
from src.structured_text_mapping import StructuredJobMatcher

LOGGER_NAME = "rolearadar.pipeline"


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SiteSummary:
    """Captures intermediate artefacts for each processed site."""

    name: str
    url_file: Optional[Path] = None
    raw_url_count: int = 0
    filtered_url_count: int = 0
    sitemap_summary_file: Optional[Path] = None
    scraper_result: Dict[str, Any] = field(default_factory=dict)
    parsed_output: Optional[Path] = None
    parsed_jobs_count: int = 0
    processed_jobs: List[Dict[str, Any]] = field(default_factory=list)
    parsing_stats: Optional[ProcessingStats] = None


@dataclass
class MatchingSummary:
    """Holds artefacts from the structured matcher."""

    total_jobs: int
    filtered_jobs: int
    min_score: float
    csv_path: Optional[Path] = None
    json_path: Optional[Path] = None
    top_matches: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[Path]:
    """Resolve a possibly-relative path against the provided base directory."""
    if not path_value:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def compile_pattern(pattern: Any) -> Any:
    """Compile include/exclude patterns supporting regex or substring rules."""
    if pattern is None:
        return None
    if isinstance(pattern, re.Pattern):
        return pattern
    if isinstance(pattern, dict) and "regex" in pattern:
        flags = 0
        if pattern.get("ignore_case", True):
            flags |= re.IGNORECASE
        if pattern.get("multiline"):
            flags |= re.MULTILINE
        return re.compile(pattern["regex"], flags)
    if isinstance(pattern, str):
        if pattern.startswith("regex:"):
            expr = pattern.split(":", 1)[1]
            return re.compile(expr, re.IGNORECASE)
        return pattern.lower()
    return str(pattern).lower()


def matches_pattern(url: str, compiled_pattern: Any) -> bool:
    """Return True if URL satisfies compiled pattern."""
    if compiled_pattern is None:
        return False
    if hasattr(compiled_pattern, "search"):
        return bool(compiled_pattern.search(url))
    return compiled_pattern in url.lower()


def filter_urls(
    urls: Iterable[str],
    include_patterns: Sequence[Any] | None = None,
    exclude_patterns: Sequence[Any] | None = None,
    allow_domains: Sequence[str] | None = None,
) -> List[str]:
    """Filter URL collection using include/exclude rules."""
    include_compiled = [compile_pattern(p) for p in include_patterns or []]
    exclude_compiled = [compile_pattern(p) for p in exclude_patterns or []]
    allowed_domains = {d.lower() for d in (allow_domains or []) if d}

    filtered: List[str] = []
    for url in urls:
        stripped = url.strip()
        if not stripped:
            continue
        normalized = stripped.lower()
        if allowed_domains:
            domain = urlparse(stripped).netloc.lower()
            if domain not in allowed_domains:
                continue
        if include_compiled and not any(matches_pattern(normalized, pat) for pat in include_compiled):
            continue
        if exclude_compiled and any(matches_pattern(normalized, pat) for pat in exclude_compiled):
            continue
        filtered.append(stripped)
    return sorted(set(filtered))


# ---------------------------------------------------------------------------
# Custom scraper implementation (pattern-driven)
# ---------------------------------------------------------------------------

class PatternJobScraper(BaseJobScraper):
    """Configurable scraper that filters job URLs by include/exclude patterns."""

    def __init__(
        self,
        url_file_path: str,
        include_patterns: Sequence[Any] | None,
        exclude_patterns: Sequence[Any] | None,
        *,
        company_name: str,
        config: ScrapingConfig,
        max_urls: Optional[int] = None,
    ) -> None:
        self._include_patterns = [compile_pattern(p) for p in (include_patterns or [])]
        self._exclude_patterns = [compile_pattern(p) for p in (exclude_patterns or [])]
        super().__init__(
            url_file_path,
            config=config,
            max_urls=max_urls,
            company_name=company_name,
        )

    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        if not self._include_patterns and not self._exclude_patterns:
            return [u for u in urls if u]
        filtered: List[str] = []
        for url in urls:
            normalized = url.lower()
            if self._include_patterns and not any(
                matches_pattern(normalized, pattern) for pattern in self._include_patterns
            ):
                continue
            if self._exclude_patterns and any(
                matches_pattern(normalized, pattern) for pattern in self._exclude_patterns
            ):
                continue
            filtered.append(url)
        return filtered


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Coordinates the end-to-end workflow described in the YAML config."""

    def __init__(self, config_path: Path, config: Dict[str, Any]) -> None:
        self.config_path = config_path
        self.config_dir = config_path.parent.resolve()
        self.repo_root = Path(__file__).resolve().parent
        self.config = config

        runtime_cfg = config.get("runtime", {})
        run_id = runtime_cfg.get("run_id") or datetime.now().strftime("%Y%m%d-%H%M%S")
        base_output = runtime_cfg.get("output_dir", "runtime_data/pipeline_runs")
        output_root = resolve_path(base_output, self.repo_root)
        if output_root is None:
            raise ValueError("Runtime output_dir could not be resolved")
        self.run_dir = ensure_dir(output_root / run_id)
        self.sites_dir = ensure_dir(self.run_dir / "sites")
        self.match_dir = ensure_dir(self.run_dir / "matches")

        self.logger = self._configure_logging(runtime_cfg)
        self.sitemap_processor_cls = self._load_sitemap_processor()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _configure_logging(self, runtime_cfg: Dict[str, Any]) -> logging.Logger:
        log_cfg = self.config.get("logging", {})
        level_name = log_cfg.get("level", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        log_filename = log_cfg.get("file", "pipeline.log")
        log_path = self.run_dir / log_filename

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logger = logging.getLogger(LOGGER_NAME)
        logger.debug("Logging configured at level %s", level_name)
        return logger

    def _load_sitemap_processor(self):
        sitemap_path = self.repo_root / "sitemap-extract" / "sitemap_extract.py"
        if not sitemap_path.exists():
            self.logger.warning("sitemap_extract.py not found at %s", sitemap_path)
            return None
        spec = importlib.util.spec_from_file_location("sitemap_extract_module", sitemap_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return getattr(module, "HumanizedSitemapProcessor", None)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        self.logger.info("Starting RoleRadar pipeline with config: %s", self.config_path)
        load_dotenv()
        self._apply_env_overrides()

        resume_payload = self._process_resume()
        site_summaries: List[SiteSummary] = []
        all_parsed_jobs: List[Dict[str, Any]] = []

        sites_cfg = self.config.get("sites", [])
        if not sites_cfg:
            self.logger.warning("No sites defined in configuration; skipping site processing")
        for site_cfg in sites_cfg:
            summary = await self._process_site(site_cfg)
            site_summaries.append(summary)
            all_parsed_jobs.extend(summary.processed_jobs)

        matching_summary = None
        if all_parsed_jobs:
            matching_summary = self._run_matching(resume_payload["resume_json"], all_parsed_jobs)
        else:
            self.logger.warning("No parsed jobs available; skipping matching stage")

        run_summary = self._build_summary(resume_payload, site_summaries, matching_summary)
        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)
        self.logger.info("Run summary written to %s", summary_path)

        return run_summary

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _apply_env_overrides(self) -> None:
        secrets = self.config.get("secrets", {}) or {}
        for key, value in secrets.items():
            if value is None:
                continue
            os.environ[str(key)] = str(value)
            self.logger.debug("Environment override set for %s", key)

    def _process_resume(self) -> Dict[str, Any]:
        resume_cfg = self.config.get("resume", {})
        resume_path = resolve_path(resume_cfg.get("path"), self.config_dir)
        if not resume_path:
            raise ValueError("Resume path must be defined in the configuration")
        use_llm = bool(resume_cfg.get("use_llm", True))
        self.logger.info("Parsing resume from %s (LLM=%s)", resume_path, use_llm)

        resume = parse_resume(str(resume_path), use_llm=use_llm)
        resume_json = asdict(resume)

        resume_out = resolve_path(resume_cfg.get("output"), self.config_dir)
        if resume_out is None:
            resume_out = self.run_dir / "resume.json"
        ensure_dir(resume_out.parent)
        save_resume_json(resume, str(resume_out))

        return {
            "resume_json": resume_json,
            "resume_output": resume_out,
        }

    async def _process_site(self, site_cfg: Dict[str, Any]) -> SiteSummary:
        name = site_cfg.get("name")
        if not name:
            raise ValueError("Each site entry requires a 'name'")
        self.logger.info("Processing site '%s'", name)

        site_dir = ensure_dir(self.sites_dir / name)
        summary = SiteSummary(name=name)

        # Stage 1: URL preparation
        urls_artifact = self._prepare_site_urls(site_cfg, site_dir)
        summary.url_file = urls_artifact.get("url_file")
        summary.raw_url_count = urls_artifact.get("raw_count", 0)
        summary.filtered_url_count = urls_artifact.get("filtered_count", 0)
        summary.sitemap_summary_file = urls_artifact.get("summary_file")

        if not summary.url_file or summary.filtered_url_count == 0:
            self.logger.warning("No URLs to scrape for site '%s'; skipping downstream stages", name)
            return summary

        # Stage 2: Scraping
        scraper_result = await self._scrape_site(site_cfg, site_dir, summary.url_file)
        summary.scraper_result = scraper_result or {}
        llm_input_file = Path(scraper_result["llm_file"]) if scraper_result and scraper_result.get("llm_file") else None
        if not llm_input_file or not llm_input_file.exists():
            self.logger.warning("Scraper did not produce LLM input for site '%s'", name)
            return summary

        # Stage 3: Job parsing via LLM
        parsed_jobs, parsing_stats, parsed_output = await self._parse_jobs(site_cfg, site_dir, llm_input_file)
        summary.processed_jobs = parsed_jobs
        summary.parsed_jobs_count = len(parsed_jobs)
        summary.parsing_stats = parsing_stats
        summary.parsed_output = parsed_output

        self.logger.info(
            "Site '%s': %d/%d scraped URLs parsed successfully",
            name,
            summary.parsed_jobs_count,
            summary.filtered_url_count,
        )
        return summary

    def _prepare_site_urls(self, site_cfg: Dict[str, Any], site_dir: Path) -> Dict[str, Any]:
        urls_cfg = site_cfg.get("urls", {})
        filters_cfg = site_cfg.get("url_filters", {}) or {}
        include_patterns = filters_cfg.get("include") or []
        exclude_patterns = filters_cfg.get("exclude") or []
        allowed_domains = filters_cfg.get("allow_domains") or []

        raw_urls: List[str] = []
        summary_artifacts: Dict[str, Any] = {}

        # Option 1: direct list of URLs
        if isinstance(urls_cfg, dict) and urls_cfg.get("list"):
            raw_urls.extend(urls_cfg.get("list") or [])
        # Option 2: existing file
        file_ref = urls_cfg.get("file") if isinstance(urls_cfg, dict) else None
        if file_ref:
            path = resolve_path(file_ref, self.config_dir)
            if path and path.exists():
                raw_urls.extend(path.read_text(encoding="utf-8").splitlines())
                summary_artifacts["source_file"] = path
            else:
                self.logger.warning("Configured URL file %s not found", file_ref)
        # Option 3: sitemap extraction
        sitemap_url = urls_cfg.get("sitemap_url") if isinstance(urls_cfg, dict) else None
        sitemap_urls = urls_cfg.get("sitemap_urls") if isinstance(urls_cfg, dict) else None
        if sitemap_url or sitemap_urls:
            extracted = self._run_sitemap_extraction(
                name=site_cfg.get("name", "site"),
                sitemap_sources=[sitemap_url] if sitemap_url else sitemap_urls,
                extractor_cfg=site_cfg.get("extractor", {}),
                site_dir=site_dir,
            )
            raw_urls.extend(extracted.get("urls", []))
            summary_artifacts.update(extracted)

        raw_urls = [u for u in raw_urls if u]
        raw_count = len(raw_urls)
        filtered_urls = filter_urls(raw_urls, include_patterns, exclude_patterns, allowed_domains)
        filtered_count = len(filtered_urls)
        if filtered_count == 0:
            self.logger.warning("URL filter removed all %d URLs for site '%s'", raw_count, site_cfg.get("name"))

        url_file = site_dir / "job_urls.txt"
        with open(url_file, "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_urls))
            f.write("\n")

        summary_artifacts.update(
            {
                "url_file": url_file,
                "raw_count": raw_count,
                "filtered_count": filtered_count,
            }
        )
        return summary_artifacts

    def _run_sitemap_extraction(
        self,
        *,
        name: str,
        sitemap_sources: Optional[Sequence[str]],
        extractor_cfg: Dict[str, Any],
        site_dir: Path,
    ) -> Dict[str, Any]:
        if not sitemap_sources:
            return {}
        if not self.sitemap_processor_cls:
            self.logger.error("Sitemap processor unavailable; skipping extraction for %s", name)
            return {}

        save_dir = ensure_dir(site_dir / "sitemaps")
        min_delay = extractor_cfg.get("min_delay", 3.0)
        max_delay = extractor_cfg.get("max_delay", 8.0)
        max_retries = extractor_cfg.get("max_retries", 3)
        max_workers = extractor_cfg.get("max_workers", 1)
        proxy_file = resolve_path(extractor_cfg.get("proxy_file"), self.config_dir)
        ua_file = resolve_path(extractor_cfg.get("user_agent_file"), self.config_dir)

        processor = self.sitemap_processor_cls(
            use_cloudscraper=not extractor_cfg.get("no_cloudscraper", False),
            proxy_file=str(proxy_file) if proxy_file else None,
            user_agent_file=str(ua_file) if ua_file else None,
            min_delay=min_delay,
            max_delay=max_delay,
            max_retries=max_retries,
            max_workers=max_workers,
            save_dir=str(save_dir),
        )

        self.logger.info("Extracting sitemaps for %s (sources=%d)", name, len(sitemap_sources))
        all_sitemaps, all_pages = processor.process_all_sitemaps(sitemap_sources)
        processor.print_summary(all_sitemaps, all_pages)

        summary_file = save_dir / "extraction_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sitemap_sources": list(all_sitemaps),
                    "page_count": len(all_pages),
                },
                f,
                indent=2,
            )

        return {
            "urls": list(all_pages),
            "summary_file": summary_file,
        }

    async def _scrape_site(self, site_cfg: Dict[str, Any], site_dir: Path, url_file: Path) -> Optional[Dict[str, Any]]:
        scraper_cfg = site_cfg.get("scraper", {}) or {}
        output_dir = ensure_dir(site_dir / "scraped")
        scraping_config = ScrapingConfig(
            max_concurrent=scraper_cfg.get("max_concurrent", 6),
            delay_between_requests=scraper_cfg.get("delay_between_requests", 0.75),
            max_retries=scraper_cfg.get("max_retries", 3),
            timeout_seconds=scraper_cfg.get("timeout_seconds", 30),
            output_dir=str(output_dir),
            max_chars=scraper_cfg.get("max_chars"),
        )
        bot_delays = scraper_cfg.get("bot_retry_delays")
        if bot_delays:
            scraping_config.bot_retry_delays = [float(v) for v in bot_delays]
        if "max_bot_retries" in scraper_cfg:
            scraping_config.max_bot_retries = int(scraper_cfg.get("max_bot_retries", 3))
        if "user_agent_rotation" in scraper_cfg:
            scraping_config.user_agent_rotation = bool(scraper_cfg.get("user_agent_rotation"))
        if "randomize_delays" in scraper_cfg:
            scraping_config.randomize_delays = bool(scraper_cfg.get("randomize_delays"))

        max_urls = scraper_cfg.get("max_urls")
        company_key = scraper_cfg.get("company_key")

        if company_key:
            self.logger.info(
                "Scraping %s using predefined scraper '%s'", site_cfg.get("name"), company_key
            )
            return await scrape_company(
                company_key,
                str(url_file),
                scraping_config,
                max_urls=max_urls,
            )

        include_patterns = scraper_cfg.get("include_patterns")
        exclude_patterns = scraper_cfg.get("exclude_patterns")
        if not include_patterns and not exclude_patterns:
            self.logger.warning(
                "Site '%s' uses custom scraper without patterns; all URLs will be scraped",
                site_cfg.get("name"),
            )

        scraper = PatternJobScraper(
            str(url_file),
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            company_name=site_cfg.get("name", "custom"),
            config=scraping_config,
            max_urls=max_urls,
        )
        return await scraper.scrape_all()

    async def _parse_jobs(
        self,
        site_cfg: Dict[str, Any],
        site_dir: Path,
        llm_input_file: Path,
    ) -> Tuple[List[Dict[str, Any]], ProcessingStats, Optional[Path]]:
        parser_cfg = site_cfg.get("parser", {}) or {}
        output_dir = ensure_dir(site_dir / "parsed")
        output_filename = parser_cfg.get("output_filename") or f"{site_cfg.get('name')}_processed.json"

        processor = AsyncJobProcessor(
            max_concurrent_requests=parser_cfg.get("max_concurrent_requests", 5),
            batch_size=parser_cfg.get("batch_size", 20),
            retry_delays=[float(d) for d in parser_cfg.get("retry_delays", [1.0, 2.0, 4.0])],
            output_dir=str(output_dir),
        )

        processed_jobs, stats = await processor.process_from_file_async(
            str(llm_input_file),
            output_filename=output_filename,
        )
        final_output = output_dir / output_filename
        return processed_jobs, stats, final_output

    def _run_matching(
        self,
        resume_json: Dict[str, Any],
        jobs: List[Dict[str, Any]],
    ) -> MatchingSummary:
        cfg = self.config.get("matching", {}) or {}
        if not cfg.get("enabled", True):
            self.logger.info("Matching disabled via configuration")
            return MatchingSummary(total_jobs=len(jobs), filtered_jobs=0, min_score=0.0)

        min_score = float(cfg.get("min_score", 0.0))
        model_name = cfg.get("model_name", "all-MiniLM-L6-v2")
        batch_size = int(cfg.get("batch_size", 100))
        self.logger.info(
            "Running structured matcher on %d jobs (model=%s, min_score=%s)",
            len(jobs),
            model_name,
            min_score,
        )

        matcher = StructuredJobMatcher(model_name=model_name, batch_size=batch_size)
        results = matcher.process_job_batch_structured(resume_json, jobs)
        filtered_results = [r for r in results if r.match_score >= min_score]

        csv_filename = cfg.get("csv_filename", "matches_detailed.csv")
        csv_path = self.match_dir / csv_filename
        matcher.export_detailed_results(filtered_results, str(csv_path))

        json_filename = cfg.get("json_filename", "matches.json")
        json_path = self.match_dir / json_filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([self._serialize_match(r) for r in filtered_results], f, indent=2)

        top_n = int(cfg.get("top_k", 5))
        top_matches = [self._serialize_match(r) for r in filtered_results[:top_n]]

        self.logger.info(
            "Matching complete: %d/%d jobs >= %.2f saved to %s",
            len(filtered_results),
            len(results),
            min_score,
            csv_path,
        )

        return MatchingSummary(
            total_jobs=len(results),
            filtered_jobs=len(filtered_results),
            min_score=min_score,
            csv_path=csv_path,
            json_path=json_path,
            top_matches=top_matches,
        )

    @staticmethod
    def _serialize_match(result) -> Dict[str, Any]:
        breakdown = result.match_breakdown
        return {
            "job_title": result.job_title,
            "company": result.company,
            "match_score": result.match_score,
            "job_url": result.job_url,
            "salary_range": result.salary_range,
            "location": result.location,
            "recommendation": result.recommendation,
            "experience_match": breakdown.experience_match,
            "skill_match": breakdown.skill_match,
            "education_match": breakdown.education_match,
            "semantic_match": breakdown.semantic_match,
            "missing_requirements": breakdown.missing_requirements,
            "improvement_suggestions": breakdown.improvement_suggestions,
        }

    def _build_summary(
        self,
        resume_payload: Dict[str, Any],
        site_summaries: List[SiteSummary],
        matching_summary: Optional[MatchingSummary],
    ) -> Dict[str, Any]:
        def path_or_none(p: Optional[Path]) -> Optional[str]:
            return str(p) if p else None

        sites_payload: List[Dict[str, Any]] = []
        for summary in site_summaries:
            stats = summary.parsing_stats
            sites_payload.append(
                {
                    "name": summary.name,
                    "url_file": path_or_none(summary.url_file),
                    "raw_url_count": summary.raw_url_count,
                    "filtered_url_count": summary.filtered_url_count,
                    "scraper_output": summary.scraper_result,
                    "parsed_output": path_or_none(summary.parsed_output),
                    "parsed_jobs_count": summary.parsed_jobs_count,
                    "parsing_stats": {
                        "total_jobs": stats.total_jobs if stats else 0,
                        "successful_parses": stats.successful_parses if stats else 0,
                        "failed_parses": stats.failed_parses if stats else 0,
                        "success_rate": stats.success_rate if stats else 0.0,
                        "processing_time": stats.processing_time if stats else 0.0,
                    } if stats else None,
                }
            )

        summary_payload = {
            "run_directory": str(self.run_dir),
            "resume_output": str(resume_payload.get("resume_output")),
            "sites": sites_payload,
        }

        if matching_summary:
            summary_payload["matching"] = {
                "total_jobs": matching_summary.total_jobs,
                "filtered_jobs": matching_summary.filtered_jobs,
                "min_score": matching_summary.min_score,
                "csv_path": path_or_none(matching_summary.csv_path),
                "json_path": path_or_none(matching_summary.json_path),
                "top_matches": matching_summary.top_matches,
            }

        return summary_payload


# ---------------------------------------------------------------------------
# Command-line entrypoint
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RoleRadar ML pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    return parser.parse_args(argv)


async def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = load_config(config_path)

    runner = PipelineRunner(config_path, config)
    run_summary = await runner.run()
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Pipeline finished. Top matches: %s", run_summary.get("matching", {}).get("top_matches"))


if __name__ == "__main__":
    asyncio.run(main())
