"""
Job processor for converting scraped markdown to structured JSON.

This module handles batch processing of scraped job postings, converting
markdown content to structured JSON format using LLM providers.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

from llm_providers import LLMProvider, get_default_provider

logger = logging.getLogger(__name__)



@dataclass
class ScrapedJob:
    """Container for scraped job data."""
    url: str
    markdown: str
    success: bool
    original_length: int
    cleaned_length: int
    scrape_timestamp: Optional[str] = None
    company_domain: Optional[str] = None


@dataclass
class ProcessingStats:
    """Statistics for job processing batch."""
    total_jobs: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    skipped_jobs: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.successful_parses / self.total_jobs) * 100
    
    @property
    def processing_time(self) -> float:
        """Calculate total processing time in seconds."""
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


class JobProcessor:
    """Process scraped job postings into structured JSON format."""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        batch_size: int = 10,
        delay_between_requests: float = 1.0,
        max_retries: int = 2,
        output_dir: str = "processed_jobs"
    ):
        """
        Initialize job processor.
        
        Args:
            llm_provider: LLM provider instance (uses default if None)
            batch_size: Number of jobs to process before saving checkpoint
            delay_between_requests: Delay between API calls (rate limiting)
            max_retries: Maximum retry attempts for failed parsing
            output_dir: Directory to save processed jobs
        """
        self.llm_provider = llm_provider or get_default_provider()
        self.batch_size = batch_size
        self.delay_between_requests = delay_between_requests
        self.max_retries = max_retries
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Processing state
        self.processed_jobs: List[Dict] = []
        self.stats = ProcessingStats()
        
        logger.info(f"JobProcessor initialized with {type(self.llm_provider).__name__}")
    
    def load_scraped_jobs(self, scraped_data_file: str) -> List[ScrapedJob]:
        """
        Load scraped job data from JSON file.
        Handles multiple formats:
        1. Simple format: {"url": "...", "markdown": "...", "company": "..."}
        2. Detailed format: {"url": "...", "markdown": "...", "success": true, "original_length": 123, ...}
        
        Args:
            scraped_data_file: Path to JSON file with scraped job data
            
        Returns:
            List of ScrapedJob objects
        """
        try:
            with open(scraped_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            jobs = []
            for item in data:
                if isinstance(item, dict) and 'markdown' in item:
                    # Handle both simple and detailed formats
                    markdown = item.get('markdown', '')
                    
                    # For simple format (your current data), assume success if markdown exists
                    if 'success' not in item:
                        success = bool(markdown and markdown.strip())
                        original_length = len(markdown) if markdown else 0
                        cleaned_length = len(markdown) if markdown else 0
                    else:
                        # Detailed format
                        success = item.get('success', False)
                        original_length = item.get('original_length', len(markdown))
                        cleaned_length = item.get('cleaned_length', len(markdown))
                    
                    job = ScrapedJob(
                        url=item.get('url', ''),
                        markdown=markdown,
                        success=success,
                        original_length=original_length,
                        cleaned_length=cleaned_length,
                        scrape_timestamp=item.get('timestamp'),
                        company_domain=item.get('company') or item.get('domain')  # Handle both 'company' and 'domain'
                    )
                    jobs.append(job)
            
            logger.info(f"Loaded {len(jobs)} scraped jobs from {scraped_data_file}")
            return jobs
            
        except Exception as exc:
            logger.error(f"Failed to load scraped jobs: {exc}")
            raise
    
    def _clean_markdown(self, markdown: str) -> str:
        """
        Clean markdown content by removing URLs and excessive whitespace.
        
        Args:
            markdown: Raw markdown content
            
        Returns:
            Cleaned markdown content
        """
        lines = markdown.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are primarily URLs or hyperlinks
            if line.strip().startswith('http') or '[' in line and '](' in line:
                continue
            
            # Remove excessive whitespace
            cleaned_line = ' '.join(line.split())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _parse_single_job(self, job: ScrapedJob, retry_count: int = 0) -> Optional[Dict]:
        """
        Parse a single job posting.
        
        Args:
            job: ScrapedJob instance to parse
            retry_count: Current retry attempt
            
        Returns:
            Parsed job dictionary or None if failed
        """
        if not job.success or not job.markdown.strip():
            logger.warning(f"Skipping job {job.url}: no content or scraping failed")
            return None
        
        try:
            # Clean the markdown
            cleaned_markdown = self._clean_markdown(job.markdown)
            
            if len(cleaned_markdown) < 100:  # Too short to be useful
                logger.warning(f"Skipping job {job.url}: content too short after cleaning")
                return None
            
            # Parse with LLM
            parsed_job = self.llm_provider.parse_job_markdown(cleaned_markdown, job.url)
            
            # Add processing metadata
            parsed_job["processing_metadata"] = {
                "original_length": job.original_length,
                "cleaned_length": len(cleaned_markdown),
                "scrape_timestamp": job.scrape_timestamp,
                "processing_timestamp": datetime.now().isoformat(),
                "retry_count": retry_count,
                "company_domain": job.company_domain
            }
            
            return parsed_job
            
        except Exception as exc:
            logger.error(f"Failed to parse job {job.url} (attempt {retry_count + 1}): {exc}")
            
            if retry_count < self.max_retries:
                logger.info(f"Retrying job {job.url} in {self.delay_between_requests * 2} seconds...")
                time.sleep(self.delay_between_requests * 2)
                return self._parse_single_job(job, retry_count + 1)
            
            return None
    
    def process_jobs(
        self,
        scraped_jobs: List[ScrapedJob],
        output_filename: str = "processed_jobs.json",
        save_checkpoints: bool = True
    ) -> Tuple[List[Dict], ProcessingStats]:
        """
        Process a list of scraped jobs into structured format.
        
        Args:
            scraped_jobs: List of ScrapedJob instances
            output_filename: Output file name for processed jobs
            save_checkpoints: Whether to save progress at regular intervals
            
        Returns:
            Tuple of (processed_jobs_list, processing_stats)
        """
        self.stats = ProcessingStats(
            total_jobs=len(scraped_jobs),
            start_time=datetime.now()
        )
        
        logger.info(f"Starting processing of {len(scraped_jobs)} jobs")
        
        for i, job in enumerate(scraped_jobs):
            logger.info(f"Processing job {i+1}/{len(scraped_jobs)}: {job.url}")
            
            # Parse the job
            parsed_job = self._parse_single_job(job)
            
            if parsed_job:
                self.processed_jobs.append(parsed_job)
                self.stats.successful_parses += 1
                logger.info(f"✓ Successfully parsed: {parsed_job['job_info']['title']}")
            else:
                self.stats.failed_parses += 1
                logger.warning(f"✗ Failed to parse job: {job.url}")
            
            # Rate limiting
            if i < len(scraped_jobs) - 1:  # Don't sleep after last job
                time.sleep(self.delay_between_requests)
            
            # Save checkpoint
            if save_checkpoints and (i + 1) % self.batch_size == 0:
                checkpoint_file = f"checkpoint_{i+1}_{output_filename}"
                self._save_jobs(checkpoint_file)
                logger.info(f"Saved checkpoint: {checkpoint_file}")
        
        self.stats.end_time = datetime.now()
        
        # Save final results
        final_output = self.output_dir / output_filename
        self._save_jobs(str(final_output))
        
        # Log summary
        logger.info(f"Processing complete!")
        logger.info(f"Success rate: {self.stats.success_rate:.1f}%")
        logger.info(f"Processing time: {self.stats.processing_time:.1f} seconds")
        logger.info(f"Output saved to: {final_output}")
        
        return self.processed_jobs, self.stats
    
    def _save_jobs(self, filename: str):
        """Save processed jobs to JSON file."""
        output_path = self.output_dir / filename
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_jobs, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Failed to save jobs to {output_path}: {exc}")
            raise
    
    def process_from_file(
        self,
        scraped_data_file: str,
        output_filename: str = "processed_jobs.json"
    ) -> Tuple[List[Dict], ProcessingStats]:
        """
        Convenience method to process jobs directly from scraped data file.
        
        Args:
            scraped_data_file: Path to scraped job data JSON file
            output_filename: Output filename for processed jobs
            
        Returns:
            Tuple of (processed_jobs_list, processing_stats)
        """
        scraped_jobs = self.load_scraped_jobs(scraped_data_file)
        return self.process_jobs(scraped_jobs, output_filename)
    
    def create_processing_report(self) -> Dict:
        """Create a detailed processing report."""
        if not self.stats.start_time:
            return {"error": "No processing has been performed yet"}
        
        # Analyze job distribution
        companies = {}
        locations = {}
        job_types = {}
        
        for job in self.processed_jobs:
            company = job["job_info"]["company"]
            location = job["job_info"]["location"]
            job_type = job["job_info"]["employment_type"]
            
            companies[company] = companies.get(company, 0) + 1
            locations[location] = locations.get(location, 0) + 1
            job_types[job_type] = job_types.get(job_type, 0) + 1
        
        return {
            "processing_summary": {
                "total_jobs_processed": self.stats.total_jobs,
                "successful_parses": self.stats.successful_parses,
                "failed_parses": self.stats.failed_parses,
                "success_rate_percent": round(self.stats.success_rate, 2),
                "processing_time_seconds": round(self.stats.processing_time, 2)
            },
            "job_distribution": {
                "top_companies": dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]),
                "top_locations": dict(sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10]),
                "employment_types": job_types
            },
            "timestamps": {
                "start_time": self.stats.start_time.isoformat() if self.stats.start_time else None,
                "end_time": self.stats.end_time.isoformat() if self.stats.end_time else None
            }
        }


# Example usage with hardcoded configuration
if __name__ == "__main__":
    load_dotenv()
    # Hardcoded configuration dictionary
    config = {
        "input_file": "/home/jd/proj/RoleRadar/src/job_data/netflix_jobs_1759030465_for_llm.json",
        "output_filename": "netflix_processed.json", 
        "batch_size": 10,
        "delay_between_requests": 1.5,
        "max_retries": 3,
        "output_dir": "processed_jobs",
        "generate_report": True
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Process jobs using config dictionary
    processor = JobProcessor(
        batch_size=config["batch_size"],
        delay_between_requests=config["delay_between_requests"],
        max_retries=config["max_retries"],
        output_dir=config["output_dir"]
    )
    
    try:
        processed_jobs, stats = processor.process_from_file(
            config["input_file"], 
            config["output_filename"]
        )
        
        if config["generate_report"]:
            report = processor.create_processing_report()
            report_file = Path(config["output_dir"]) / "processing_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Processing report saved to: {report_file}")
        
        print(f"Successfully processed {stats.successful_parses}/{stats.total_jobs} jobs")
        print(f"Success rate: {stats.success_rate:.1f}%")
        print(f"Processing time: {stats.processing_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)