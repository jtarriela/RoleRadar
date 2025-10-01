#!/usr/bin/env python3
"""
RoleRadar CLI - YAML Configuration-Based Job Scraping and Matching

This CLI tool orchestrates the complete job scraping and matching pipeline
using a YAML configuration file for all parameters.

Usage:
    python cli.py --config config.yaml scrape --company meta
    python cli.py --config config.yaml process --input job_data/meta.json
    python cli.py --config config.yaml parse-resume --file resume.pdf
    python cli.py --config config.yaml match
    python cli.py --config config.yaml run-all --company meta --resume resume.pdf
"""

import os
import sys
import argparse
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate YAML configuration."""
    
    def __init__(self, config_path: str):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._setup_environment()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            logger.info("Create a config file using config.example.yaml as template")
            sys.exit(1)
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config or {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            sys.exit(1)
    
    def _validate_config(self):
        """Validate required configuration sections."""
        required_sections = ['scraper', 'job_processor', 'matcher', 'llm', 'output']
        missing = [s for s in required_sections if s not in self.config]
        
        if missing:
            logger.warning(f"Missing configuration sections: {', '.join(missing)}")
            logger.info("Using default values for missing sections")
    
    def _setup_environment(self):
        """Set up environment variables from configuration."""
        # Set LLM provider from config if not already set
        llm_config = self.config.get('llm', {})
        
        if 'provider' in llm_config and not os.getenv('LLM_PROVIDER'):
            os.environ['LLM_PROVIDER'] = llm_config['provider']
        
        # Set model if specified
        if 'gemini_model' in llm_config and not os.getenv('GEMINI_MODEL'):
            os.environ['GEMINI_MODEL'] = llm_config['gemini_model']
        
        if 'openai_model' in llm_config and not os.getenv('OPENAI_MODEL'):
            os.environ['OPENAI_MODEL'] = llm_config['openai_model']
    
    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section name
            key: Key within section (optional)
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        if key is None:
            return self.config.get(section, default)
        
        section_config = self.config.get(section, {})
        return section_config.get(key, default)


class RoleRadarCLI:
    """Main CLI class for RoleRadar."""
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize CLI.
        
        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create output directories if they don't exist."""
        output_config = self.config.get('output', {})
        
        dirs_to_create = [
            output_config.get('base_dir', 'runtime_data'),
            output_config.get('sitemap_dir', 'runtime_data/sitemaps'),
            output_config.get('scraped_dir', 'runtime_data/scraped_jobs'),
            output_config.get('processed_dir', 'runtime_data/processed_job_data'),
            output_config.get('resume_dir', 'runtime_data/processed_resumes'),
            output_config.get('matches_dir', 'runtime_data/match_results'),
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def scrape_jobs(self, company: str = None, url_file: str = None):
        """
        Scrape job postings from company website.
        
        Args:
            company: Company name (from config.companies)
            url_file: Path to text file with URLs (alternative to company)
        """
        from raw_job_scraper import scrape_company, ScrapingConfig
        
        logger.info("=" * 80)
        logger.info("STEP 1: SCRAPING JOB POSTINGS")
        logger.info("=" * 80)
        
        # Get scraper configuration
        scraper_config = self.config.get('scraper', {})
        
        # Create ScrapingConfig from YAML
        config = ScrapingConfig(
            max_concurrent=scraper_config.get('max_concurrent', 6),
            delay_between_requests=scraper_config.get('delay_between_requests', 1.0),
            max_retries=scraper_config.get('max_retries', 3),
            timeout_seconds=scraper_config.get('timeout_seconds', 30),
            output_dir=scraper_config.get('output_dir', 'job_data'),
            max_chars=scraper_config.get('max_chars'),
            user_agent_rotation=scraper_config.get('user_agent_rotation', True),
            randomize_delays=scraper_config.get('randomize_delays', True),
            bot_retry_delays=scraper_config.get('bot_retry_delays', [10.0, 30.0, 60.0, 120.0]),
            max_bot_retries=scraper_config.get('max_bot_retries', 3)
        )
        
        max_urls = scraper_config.get('max_urls', 10000)
        
        if url_file:
            # Scrape from URL file
            logger.info(f"Scraping jobs from URL file: {url_file}")
            result = asyncio.run(scrape_company(company or 'custom', url_file, config, max_urls))
        elif company:
            # Look up company in config
            companies = self.config.get('companies', [])
            company_config = next((c for c in companies if c.get('name') == company), None)
            
            if not company_config:
                logger.error(f"Company '{company}' not found in configuration")
                logger.info(f"Available companies: {[c.get('name') for c in companies]}")
                sys.exit(1)
            
            sitemap_url = company_config.get('sitemap_url')
            if sitemap_url:
                logger.info(f"Scraping jobs for {company} from {sitemap_url}")
                # First parse sitemap to get URLs
                # This would require sitemap parsing integration
                logger.warning("Sitemap parsing not yet integrated - please provide URL file")
                return
            else:
                logger.error(f"No sitemap_url configured for company '{company}'")
                sys.exit(1)
        else:
            logger.error("Either --company or --url-file must be provided")
            sys.exit(1)
        
        logger.info(f"✅ Scraping completed: {result}")
    
    def process_jobs(self, input_file: str, output_file: str = None):
        """
        Process scraped markdown to structured JSON.
        
        Args:
            input_file: Input JSON file with scraped markdown
            output_file: Output JSON file for processed jobs (optional)
        """
        from jobdata_markdown_to_json import JobProcessor
        
        logger.info("=" * 80)
        logger.info("STEP 2: PROCESSING JOBS TO STRUCTURED JSON")
        logger.info("=" * 80)
        
        # Get processor configuration
        processor_config = self.config.get('job_processor', {})
        
        # Create JobProcessor from YAML config
        processor = JobProcessor(
            batch_size=processor_config.get('batch_size', 10),
            delay_between_requests=processor_config.get('delay_between_requests', 1.0),
            max_retries=processor_config.get('max_retries', 2),
            output_dir=processor_config.get('output_dir', 'processed_jobs')
        )
        
        # Determine output filename
        if not output_file:
            input_path = Path(input_file)
            output_file = f"processed_{input_path.name}"
        
        logger.info(f"Processing jobs from: {input_file}")
        logger.info(f"Output will be saved to: {output_file}")
        
        # Process jobs
        processed_jobs, stats = processor.process_from_file(input_file, output_file)
        
        logger.info("=" * 80)
        logger.info(f"✅ Processing completed!")
        logger.info(f"   Total jobs: {stats.total_jobs}")
        logger.info(f"   Successful: {stats.successful_parses}")
        logger.info(f"   Failed: {stats.failed_parses}")
        logger.info(f"   Success rate: {stats.success_rate:.1f}%")
        logger.info("=" * 80)
    
    def parse_resume(self, resume_file: str, output_file: str = None):
        """
        Parse resume to structured JSON.
        
        Args:
            resume_file: Input resume file (PDF, DOCX, or TXT)
            output_file: Output JSON file (optional)
        """
        from resume_parser import parse_resume_file
        
        logger.info("=" * 80)
        logger.info("STEP 3: PARSING RESUME")
        logger.info("=" * 80)
        
        # Get resume parser configuration
        parser_config = self.config.get('resume_parser', {})
        use_llm = parser_config.get('use_llm', True)
        
        logger.info(f"Parsing resume: {resume_file}")
        logger.info(f"Using LLM: {use_llm}")
        
        # Parse resume
        resume = parse_resume_file(resume_file, use_llm=use_llm)
        
        # Determine output filename
        if not output_file:
            output_dir = parser_config.get('output_dir', 'processed_resumes')
            input_path = Path(resume_file)
            output_file = Path(output_dir) / f"{input_path.stem}.json"
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        from dataclasses import asdict
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(resume), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Resume parsed and saved to: {output_path}")
    
    def match_jobs(self, resume_path: str = None, jobs_path: str = None, output_path: str = None):
        """
        Match resume against jobs using embeddings.
        
        Args:
            resume_path: Path to resume JSON (from config if not provided)
            jobs_path: Path to jobs JSON (from config if not provided)
            output_path: Path for output CSV (from config if not provided)
        """
        from cosine_similiarity import GeminiJobMatcher, main as matcher_main
        import json
        
        logger.info("=" * 80)
        logger.info("STEP 4: MATCHING JOBS TO RESUME")
        logger.info("=" * 80)
        
        # Get matcher configuration
        matcher_config = self.config.get('matcher', {})
        
        # Use config defaults if paths not provided
        resume_path = resume_path or matcher_config.get('resume_path')
        jobs_path = jobs_path or matcher_config.get('jobs_path')
        output_path = output_path or matcher_config.get('output_path')
        
        if not resume_path or not Path(resume_path).exists():
            logger.error(f"Resume file not found: {resume_path}")
            logger.info("Parse a resume first with: cli.py parse-resume --file <resume_file>")
            sys.exit(1)
        
        if not jobs_path or not Path(jobs_path).exists():
            logger.error(f"Jobs file not found: {jobs_path}")
            logger.info("Process jobs first with: cli.py process --input <scraped_jobs.json>")
            sys.exit(1)
        
        logger.info(f"Resume: {resume_path}")
        logger.info(f"Jobs: {jobs_path}")
        logger.info(f"Output: {output_path}")
        
        # Load data
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        
        with open(jobs_path, 'r') as f:
            jobs_data = json.load(f)
        
        # Get matching parameters
        max_jobs = matcher_config.get('max_jobs')
        if max_jobs:
            jobs_data = jobs_data[:max_jobs]
            logger.info(f"Limited to first {max_jobs} jobs for testing")
        
        min_score = matcher_config.get('min_score_filter', 45.0)
        batch_size = matcher_config.get('batch_size', 50)
        
        # Initialize matcher
        try:
            matcher = GeminiJobMatcher(batch_size=batch_size)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        
        # Calculate matches
        logger.info(f"Calculating matches for {len(jobs_data)} jobs...")
        results = matcher.calculate_job_matches(resume_data, jobs_data)
        
        # Filter results
        filtered_results = matcher.filter_results(results, min_score)
        
        # Export results
        df = matcher.export_results(filtered_results, output_path)
        
        logger.info("=" * 80)
        logger.info(f"✅ Matching completed!")
        logger.info(f"   Total jobs processed: {len(jobs_data)}")
        logger.info(f"   Matches above {min_score}%: {len(filtered_results)}")
        logger.info(f"   Results saved to: {output_path}")
        logger.info("=" * 80)
    
    def run_all(self, company: str = None, resume_file: str = None, url_file: str = None):
        """
        Run complete pipeline: scrape -> process -> parse resume -> match.
        
        Args:
            company: Company to scrape
            resume_file: Resume file to parse
            url_file: URL file (alternative to company)
        """
        logger.info("=" * 80)
        logger.info("RUNNING COMPLETE PIPELINE")
        logger.info("=" * 80)
        
        workflow_config = self.config.get('workflow', {})
        
        # Step 1: Scrape jobs
        if workflow_config.get('run_scraper', True) and (company or url_file):
            self.scrape_jobs(company=company, url_file=url_file)
        
        # Step 2: Process jobs
        if workflow_config.get('run_job_processor', True):
            # Find the most recent scraped file
            scraper_config = self.config.get('scraper', {})
            output_dir = Path(scraper_config.get('output_dir', 'job_data'))
            
            # Look for company-specific file or most recent
            if company:
                scraped_file = output_dir / f"{company}.json"
            else:
                json_files = list(output_dir.glob('*.json'))
                if json_files:
                    scraped_file = max(json_files, key=lambda p: p.stat().st_mtime)
                else:
                    logger.error("No scraped job files found")
                    return
            
            if scraped_file.exists():
                self.process_jobs(str(scraped_file))
            else:
                logger.warning(f"Scraped file not found: {scraped_file}")
        
        # Step 3: Parse resume
        if workflow_config.get('run_resume_parser', False) and resume_file:
            self.parse_resume(resume_file)
        
        # Step 4: Match jobs
        if workflow_config.get('run_matcher', False):
            self.match_jobs()
        
        logger.info("=" * 80)
        logger.info("✅ PIPELINE COMPLETED")
        logger.info("=" * 80)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='RoleRadar - YAML-based job scraping and matching CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape jobs for a company
  python cli.py --config config.yaml scrape --company meta
  
  # Scrape from URL file
  python cli.py --config config.yaml scrape --url-file urls.txt
  
  # Process scraped jobs to structured JSON
  python cli.py --config config.yaml process --input job_data/meta.json
  
  # Parse resume
  python cli.py --config config.yaml parse-resume --file resume.pdf
  
  # Match jobs to resume
  python cli.py --config config.yaml match
  
  # Run complete pipeline
  python cli.py --config config.yaml run-all --company meta --resume resume.pdf
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape job postings')
    scrape_parser.add_argument('--company', type=str, help='Company name from config')
    scrape_parser.add_argument('--url-file', type=str, help='Text file with job URLs')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process scraped jobs to JSON')
    process_parser.add_argument('--input', type=str, required=True, help='Input scraped JSON file')
    process_parser.add_argument('--output', type=str, help='Output processed JSON file')
    
    # Parse resume command
    resume_parser = subparsers.add_parser('parse-resume', help='Parse resume to structured JSON')
    resume_parser.add_argument('--file', type=str, required=True, help='Resume file (PDF, DOCX, TXT)')
    resume_parser.add_argument('--output', type=str, help='Output JSON file')
    
    # Match command
    match_parser = subparsers.add_parser('match', help='Match jobs to resume')
    match_parser.add_argument('--resume', type=str, help='Resume JSON file')
    match_parser.add_argument('--jobs', type=str, help='Jobs JSON file')
    match_parser.add_argument('--output', type=str, help='Output CSV file')
    
    # Run all command
    runall_parser = subparsers.add_parser('run-all', help='Run complete pipeline')
    runall_parser.add_argument('--company', type=str, help='Company to scrape')
    runall_parser.add_argument('--url-file', type=str, help='URL file to scrape')
    runall_parser.add_argument('--resume', type=str, help='Resume file to parse')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    config = ConfigLoader(args.config)
    
    # Create CLI instance
    cli = RoleRadarCLI(config)
    
    # Execute command
    try:
        if args.command == 'scrape':
            cli.scrape_jobs(company=args.company, url_file=args.url_file)
        
        elif args.command == 'process':
            cli.process_jobs(input_file=args.input, output_file=args.output)
        
        elif args.command == 'parse-resume':
            cli.parse_resume(resume_file=args.file, output_file=args.output)
        
        elif args.command == 'match':
            cli.match_jobs(
                resume_path=args.resume,
                jobs_path=args.jobs,
                output_path=args.output
            )
        
        elif args.command == 'run-all':
            cli.run_all(
                company=args.company,
                resume_file=args.resume,
                url_file=args.url_file
            )
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
