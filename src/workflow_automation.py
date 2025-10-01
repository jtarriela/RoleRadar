#!/usr/bin/env python3
"""
Automated Job Scraping Workflow
Chains together sitemap parsing, job scraping, markdown processing, and job matching.

Usage:
    python workflow_automation.py --sitemap-url "https://example.com/sitemap.xml" --company-name "example"
    
Environment Variables:
    GEMINI_API_KEY: Your Gemini API key for LLM processing
    GEMINI_MODEL: Gemini model to use (default: gemini-flash-lite-latest)
"""

import os
import sys
import argparse
import subprocess
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))


class WorkflowAutomation:
    """Automates the complete job scraping and matching workflow."""
    
    def __init__(
        self,
        sitemap_url: str,
        company_name: str,
        output_dir: Optional[str] = None,
        resume_path: Optional[str] = None,
        skip_matching: bool = False
    ):
        """
        Initialize workflow automation.
        
        Args:
            sitemap_url: URL of the sitemap.xml to process
            company_name: Name of the company (used for file naming)
            output_dir: Directory for all outputs (default: ./workflow_output)
            resume_path: Path to resume JSON for matching (optional)
            skip_matching: Skip the final matching step
        """
        self.sitemap_url = sitemap_url
        self.company_name = company_name.lower().replace(" ", "_")
        self.skip_matching = skip_matching
        
        # Setup directories
        self.base_dir = Path(__file__).parent.parent
        self.output_dir = Path(output_dir) if output_dir else self.base_dir / "workflow_output"
        self.sitemap_dir = self.output_dir / "sitemaps"
        self.scraped_dir = self.output_dir / "scraped_jobs"
        self.processed_dir = self.output_dir / "processed_jobs"
        self.matches_dir = self.output_dir / "job_matches"
        
        # Create all directories
        for directory in [self.output_dir, self.sitemap_dir, self.scraped_dir, 
                         self.processed_dir, self.matches_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # File paths that will be generated
        self.sitemap_file = self.sitemap_dir / f"{self.company_name}_sitemap.txt"
        self.scraped_json = None  # Will be set after scraping
        self.processed_json = None  # Will be set after processing
        self.matches_csv = None  # Will be set after matching
        
        self.resume_path = resume_path
        
        # Check environment variables
        self._check_environment()
        
        # Workflow state
        self.workflow_log = []
        
    def _check_environment(self):
        """Check required environment variables."""
        if not os.getenv('GEMINI_API_KEY'):
            print("⚠️  WARNING: GEMINI_API_KEY not set. LLM processing will fail.")
            print("   Set with: export GEMINI_API_KEY='your-key-here'")
        
        if not os.getenv('GEMINI_MODEL'):
            print("ℹ️  GEMINI_MODEL not set, using default: gemini-flash-lite-latest")
            os.environ['GEMINI_MODEL'] = 'gemini-flash-lite-latest'
    
    def _log_step(self, step_num: int, step_name: str, status: str, details: str = ""):
        """Log workflow step."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "step": step_num,
            "name": step_name,
            "status": status,
            "details": details
        }
        self.workflow_log.append(log_entry)
        
        # Print to console
        status_icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⏳"
        print(f"\n{status_icon} Step {step_num}: {step_name} - {status}")
        if details:
            print(f"   {details}")
    
    def step1_parse_sitemap(self) -> bool:
        """Step 1: Parse sitemap and extract job URLs."""
        self._log_step(1, "Parse Sitemap", "RUNNING", f"URL: {self.sitemap_url}")
        
        try:
            # Path to sitemap_extract.py
            sitemap_script = self.base_dir / "sitemap-extract" / "sitemap_extract.py"
            
            if not sitemap_script.exists():
                raise FileNotFoundError(f"Sitemap script not found at {sitemap_script}")
            
            # Run sitemap extraction
            cmd = [
                "python3",
                str(sitemap_script),
                "--url", self.sitemap_url,
                "--save-dir", str(self.sitemap_dir)
            ]
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find the generated text file
            txt_files = list(self.sitemap_dir.glob("*.txt"))
            if not txt_files:
                raise FileNotFoundError("No .txt file generated from sitemap parsing")
            
            # Use the most recent file or rename it
            latest_file = max(txt_files, key=lambda p: p.stat().st_mtime)
            if latest_file != self.sitemap_file:
                latest_file.rename(self.sitemap_file)
            
            # Count URLs
            with open(self.sitemap_file, 'r') as f:
                url_count = len([line for line in f if line.strip()])
            
            self._log_step(1, "Parse Sitemap", "SUCCESS", 
                          f"Extracted {url_count} URLs to {self.sitemap_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log_step(1, "Parse Sitemap", "FAILED", f"Error: {e.stderr}")
            return False
        except Exception as e:
            self._log_step(1, "Parse Sitemap", "FAILED", str(e))
            return False
    
    def step2_scrape_jobs(self) -> bool:
        """Step 2: Scrape job postings using raw_job_scraper."""
        self._log_step(2, "Scrape Jobs", "RUNNING", f"Input: {self.sitemap_file}")
        
        try:
            # Import and run scraper
            sys.path.insert(0, str(self.base_dir / "src"))
            from raw_job_scraper import scrape_company, ScrapingConfig
            
            # Create scraping config
            config = ScrapingConfig(
                max_concurrent=6,
                delay_between_requests=1.0,
                max_chars=100000,
                output_dir=str(self.scraped_dir),
                user_agent_rotation=True,
                randomize_delays=True,
                bot_retry_delays=[10.0, 30.0, 60.0, 120.0],
                max_bot_retries=3
            )
            
            # Create a generic scraper dynamically
            from raw_job_scraper import BaseJobScraper
            
            class GenericScraper(BaseJobScraper):
                """Generic scraper that accepts all URLs from sitemap."""
                def _filter_job_urls(self, urls):
                    # Return all URLs - assume sitemap only has job URLs
                    return urls
            
            # Register the generic scraper
            from raw_job_scraper import ScraperFactory
            ScraperFactory.SCRAPERS[self.company_name] = GenericScraper
            
            # Run scraper
            async def run_scraper():
                result = await scrape_company(
                    self.company_name, 
                    str(self.sitemap_file), 
                    config
                )
                return result
            
            result = asyncio.run(run_scraper())
            
            if not result or result['successful'] == 0:
                raise Exception("No jobs scraped successfully")
            
            # Get the output file
            self.scraped_json = Path(result['llm_file'])
            
            self._log_step(2, "Scrape Jobs", "SUCCESS",
                          f"Scraped {result['successful']} jobs to {self.scraped_json}")
            return True
            
        except Exception as e:
            self._log_step(2, "Scrape Jobs", "FAILED", str(e))
            return False
    
    def step3_process_markdown(self) -> bool:
        """Step 3: Process markdown into structured JSON using LLM."""
        self._log_step(3, "Process with LLM", "RUNNING", f"Input: {self.scraped_json}")
        
        try:
            if not self.scraped_json or not self.scraped_json.exists():
                raise FileNotFoundError(f"Scraped JSON not found: {self.scraped_json}")
            
            # Import async processor
            sys.path.insert(0, str(self.base_dir / "src"))
            from jobdata_markdown_json_concurrent import AsyncJobProcessor
            
            # Create processor with moderate concurrency
            processor = AsyncJobProcessor(
                max_concurrent_requests=5,
                batch_size=20,
                output_dir=str(self.processed_dir)
            )
            
            # Generate output filename
            output_filename = f"{self.company_name}_processed.json"
            
            # Process jobs
            async def run_processor():
                processed_jobs, stats = await processor.process_from_file_async(
                    str(self.scraped_json),
                    output_filename
                )
                return processed_jobs, stats
            
            processed_jobs, stats = asyncio.run(run_processor())
            
            self.processed_json = self.processed_dir / output_filename
            
            self._log_step(3, "Process with LLM", "SUCCESS",
                          f"Processed {stats.successful_parses}/{stats.total_jobs} jobs to {self.processed_json}")
            return True
            
        except Exception as e:
            self._log_step(3, "Process with LLM", "FAILED", str(e))
            return False
    
    def step4_match_jobs(self) -> bool:
        """Step 4: Match jobs using cosine similarity."""
        if self.skip_matching:
            self._log_step(4, "Match Jobs", "SKIPPED", "Matching disabled")
            return True
        
        if not self.resume_path:
            self._log_step(4, "Match Jobs", "SKIPPED", "No resume provided")
            return True
        
        self._log_step(4, "Match Jobs", "RUNNING", 
                      f"Resume: {self.resume_path}, Jobs: {self.processed_json}")
        
        try:
            if not self.processed_json or not self.processed_json.exists():
                raise FileNotFoundError(f"Processed JSON not found: {self.processed_json}")
            
            if not Path(self.resume_path).exists():
                raise FileNotFoundError(f"Resume not found: {self.resume_path}")
            
            # Import matcher
            sys.path.insert(0, str(self.base_dir / "src"))
            from cosine_similiarity import GeminiJobMatcher
            
            # Load data
            with open(self.resume_path, 'r') as f:
                resume_data = json.load(f)
            
            with open(self.processed_json, 'r') as f:
                jobs_data = json.load(f)
            
            # Create matcher
            matcher = GeminiJobMatcher(batch_size=50)
            
            # Match jobs
            matches = matcher.match_resume_to_jobs(resume_data, jobs_data)
            
            # Save results
            output_csv = self.matches_dir / f"{self.company_name}_matches.csv"
            matcher.save_matches_to_csv(matches, str(output_csv))
            self.matches_csv = output_csv
            
            # Filter high matches
            high_matches = [m for m in matches if m.match_score >= 70.0]
            
            self._log_step(4, "Match Jobs", "SUCCESS",
                          f"Found {len(high_matches)} high matches (≥70%) out of {len(matches)} total")
            return True
            
        except Exception as e:
            self._log_step(4, "Match Jobs", "FAILED", str(e))
            return False
    
    def run(self) -> bool:
        """Run the complete workflow."""
        print("=" * 80)
        print("🚀 AUTOMATED JOB SCRAPING WORKFLOW")
        print("=" * 80)
        print(f"Company: {self.company_name}")
        print(f"Sitemap: {self.sitemap_url}")
        print(f"Output: {self.output_dir}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run each step
        steps = [
            self.step1_parse_sitemap,
            self.step2_scrape_jobs,
            self.step3_process_markdown,
            self.step4_match_jobs
        ]
        
        for step_func in steps:
            if not step_func():
                print("\n❌ Workflow failed. Check logs above.")
                self._save_log()
                return False
        
        elapsed_time = time.time() - start_time
        
        # Success summary
        print("\n" + "=" * 80)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print(f"\n📁 Output files:")
        print(f"   Sitemap URLs: {self.sitemap_file}")
        print(f"   Scraped data: {self.scraped_json}")
        print(f"   Processed jobs: {self.processed_json}")
        if self.matches_csv:
            print(f"   Job matches: {self.matches_csv}")
        print("=" * 80)
        
        self._save_log()
        return True
    
    def _save_log(self):
        """Save workflow log to file."""
        log_file = self.output_dir / f"{self.company_name}_workflow_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.workflow_log, f, indent=2)
        print(f"\n📝 Workflow log saved to: {log_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated job scraping workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - scrape and process jobs
  python workflow_automation.py --sitemap-url "https://careers.google.com/jobs/sitemap" --company-name "google"
  
  # Include job matching with resume
  python workflow_automation.py \\
    --sitemap-url "https://careers.google.com/jobs/sitemap" \\
    --company-name "google" \\
    --resume-path "../runtime_data/processed_resumes/my_resume.json"
  
  # Custom output directory
  python workflow_automation.py \\
    --sitemap-url "https://jobs.apple.com/sitemap/sitemap-jobs-en-us.xml" \\
    --company-name "apple" \\
    --output-dir "./apple_jobs"

Environment variables needed:
  GEMINI_API_KEY: Your Gemini API key
  GEMINI_MODEL: Model to use (default: gemini-flash-lite-latest)
        """
    )
    
    parser.add_argument(
        '--sitemap-url',
        required=True,
        help='URL of the sitemap.xml to process'
    )
    
    parser.add_argument(
        '--company-name',
        required=True,
        help='Name of the company (used for file naming)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Directory for all workflow outputs (default: ./workflow_output)'
    )
    
    parser.add_argument(
        '--resume-path',
        help='Path to processed resume JSON for job matching (optional)'
    )
    
    parser.add_argument(
        '--skip-matching',
        action='store_true',
        help='Skip the final job matching step'
    )
    
    args = parser.parse_args()
    
    # Create and run workflow
    workflow = WorkflowAutomation(
        sitemap_url=args.sitemap_url,
        company_name=args.company_name,
        output_dir=args.output_dir,
        resume_path=args.resume_path,
        skip_matching=args.skip_matching
    )
    
    success = workflow.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
