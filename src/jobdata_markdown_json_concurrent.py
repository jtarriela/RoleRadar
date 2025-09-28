"""
Async Job processor for parallel LLM processing.

This version uses asyncio to process multiple jobs concurrently while
respecting API rate limits and implementing proper error handling.
"""

import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import aiohttp
from asyncio import Semaphore
from dotenv import load_dotenv
# Your existing imports
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
        if self.total_jobs == 0:
            return 0.0
        return (self.successful_parses / self.total_jobs) * 100
    
    @property
    def processing_time(self) -> float:
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


class AsyncLLMProvider:
    """Async wrapper for LLM providers to enable concurrent processing."""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._semaphore = None
        
    def set_concurrency_limit(self, limit: int):
        """Set the maximum number of concurrent requests."""
        self._semaphore = Semaphore(limit)
    
    async def parse_job_markdown_async(self, markdown_content: str, job_url: str = "") -> Dict[str, object]:
        """Async wrapper for job parsing with rate limiting."""
        if self._semaphore:
            async with self._semaphore:
                # Run the sync LLM call in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    self.provider.parse_job_markdown, 
                    markdown_content, 
                    job_url
                )
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                self.provider.parse_job_markdown, 
                markdown_content, 
                job_url
            )


class AsyncJobProcessor:
    """Async job processor for parallel LLM processing."""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        max_concurrent_requests: int = 5,
        batch_size: int = 20,
        retry_delays: List[float] = [1.0, 2.0, 4.0],
        output_dir: str = "processed_jobs"
    ):
        """
        Initialize async job processor.
        
        Args:
            llm_provider: LLM provider instance
            max_concurrent_requests: Maximum parallel API calls (start with 3-5)
            batch_size: Jobs to process before saving checkpoint
            retry_delays: Delay sequence for retries (exponential backoff)
            output_dir: Directory to save processed jobs
        """
        base_provider = llm_provider or get_default_provider()
        self.llm_provider = AsyncLLMProvider(base_provider)
        self.llm_provider.set_concurrency_limit(max_concurrent_requests)
        
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_size = batch_size
        self.retry_delays = retry_delays
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Processing state
        self.processed_jobs: List[Dict] = []
        self.stats = ProcessingStats()
        
        logger.info(f"AsyncJobProcessor initialized with {max_concurrent_requests} max concurrent requests")
    
    def load_scraped_jobs(self, scraped_data_file: str) -> List[ScrapedJob]:
        """Load scraped job data from JSON file."""
        try:
            with open(scraped_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            jobs = []
            for item in data:
                if isinstance(item, dict) and 'markdown' in item:
                    markdown = item.get('markdown', '')
                    
                    if 'success' not in item:
                        success = bool(markdown and markdown.strip())
                        original_length = len(markdown) if markdown else 0
                        cleaned_length = len(markdown) if markdown else 0
                    else:
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
                        company_domain=item.get('company') or item.get('domain')
                    )
                    jobs.append(job)
            
            logger.info(f"Loaded {len(jobs)} scraped jobs from {scraped_data_file}")
            return jobs
            
        except Exception as exc:
            logger.error(f"Failed to load scraped jobs: {exc}")
            raise
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean markdown content."""
        lines = markdown.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.strip().startswith('http') or '[' in line and '](' in line:
                continue
            cleaned_line = ' '.join(line.split())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    async def _parse_single_job_async(self, job: ScrapedJob, job_index: int) -> Optional[Dict]:
        """Parse a single job with async LLM calls and retry logic."""
        if not job.success or not job.markdown.strip():
            logger.warning(f"Skipping job {job_index+1}: {job.url} - no content")
            return None
        
        cleaned_markdown = self._clean_markdown(job.markdown)
        
        if len(cleaned_markdown) < 100:
            logger.warning(f"Skipping job {job_index+1}: content too short after cleaning")
            return None
        
        # Retry logic with exponential backoff
        last_exception = None
        
        for attempt, delay in enumerate([0] + self.retry_delays):
            if delay > 0:
                logger.info(f"Retrying job {job_index+1} in {delay}s (attempt {attempt+1})")
                await asyncio.sleep(delay)
            
            try:
                parsed_job = await self.llm_provider.parse_job_markdown_async(
                    cleaned_markdown, job.url
                )
                
                # Add processing metadata
                parsed_job["processing_metadata"] = {
                    "original_length": job.original_length,
                    "cleaned_length": len(cleaned_markdown),
                    "scrape_timestamp": job.scrape_timestamp,
                    "processing_timestamp": datetime.now().isoformat(),
                    "retry_count": attempt,
                    "company_domain": job.company_domain,
                    "job_index": job_index
                }
                
                logger.info(f"✓ Parsed job {job_index+1}/{self.stats.total_jobs}: {parsed_job['job_info']['title']}")
                return parsed_job
                
            except Exception as exc:
                last_exception = exc
                logger.warning(f"Job {job_index+1} attempt {attempt+1} failed: {exc}")
                
                # Check if it's a rate limit error (API-specific handling)
                if "rate limit" in str(exc).lower() or "429" in str(exc):
                    # Add extra delay for rate limits
                    await asyncio.sleep(delay * 2)
        
        logger.error(f"✗ Failed to parse job {job_index+1} after all retries: {last_exception}")
        return None
    
    async def process_jobs_async(
        self,
        scraped_jobs: List[ScrapedJob],
        output_filename: str = "processed_jobs.json",
        save_checkpoints: bool = True
    ) -> Tuple[List[Dict], ProcessingStats]:
        """
        Process jobs asynchronously with parallel LLM calls.
        
        Args:
            scraped_jobs: List of ScrapedJob instances
            output_filename: Output file name
            save_checkpoints: Whether to save progress checkpoints
            
        Returns:
            Tuple of (processed_jobs_list, processing_stats)
        """
        self.stats = ProcessingStats(
            total_jobs=len(scraped_jobs),
            start_time=datetime.now()
        )
        
        logger.info(f"Starting async processing of {len(scraped_jobs)} jobs")
        logger.info(f"Max concurrent requests: {self.max_concurrent_requests}")
        
        # Process jobs in batches to manage memory and save checkpoints
        for batch_start in range(0, len(scraped_jobs), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(scraped_jobs))
            batch_jobs = scraped_jobs[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start+1}-{batch_end}/{len(scraped_jobs)}")
            
            # Create async tasks for the batch
            tasks = [
                self._parse_single_job_async(job, batch_start + i) 
                for i, job in enumerate(batch_jobs)
            ]
            
            # Process batch concurrently
            batch_start_time = time.time()
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_time = time.time() - batch_start_time
            
            # Process results
            successful_in_batch = 0
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    self.stats.failed_parses += 1
                elif result is not None:
                    self.processed_jobs.append(result)
                    self.stats.successful_parses += 1
                    successful_in_batch += 1
                else:
                    self.stats.failed_parses += 1
            
            logger.info(f"Batch complete: {successful_in_batch}/{len(batch_jobs)} successful")
            logger.info(f"Batch time: {batch_time:.1f}s ({len(batch_jobs)/batch_time:.1f} jobs/sec)")
            
            # Save checkpoint
            if save_checkpoints:
                checkpoint_file = f"checkpoint_batch_{batch_end}_{output_filename}"
                await self._save_jobs_async(checkpoint_file)
        
        self.stats.end_time = datetime.now()
        
        # Save final results
        final_output = self.output_dir / output_filename
        await self._save_final_jobs_async(final_output)
        
        # Log summary
        throughput = len(scraped_jobs) / self.stats.processing_time if self.stats.processing_time > 0 else 0
        logger.info(f"Async processing complete!")
        logger.info(f"Success rate: {self.stats.success_rate:.1f}%")
        logger.info(f"Total time: {self.stats.processing_time:.1f}s")
        logger.info(f"Throughput: {throughput:.1f} jobs/sec")
        logger.info(f"Output saved to: {final_output}")
        
        return self.processed_jobs, self.stats
    
    async def _save_jobs_async(self, filename: str):
        """Save processed jobs to JSON file asynchronously (for checkpoints)."""
        output_path = self.output_dir / filename
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_save, output_path)
            logger.info(f"Saved checkpoint: {output_path}")
        except Exception as exc:
            logger.error(f"Failed to save checkpoint to {output_path}: {exc}")
            raise
    
    async def _save_final_jobs_async(self, full_path: Path):
        """Save final processed jobs to specific path."""
        try:
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_save, full_path)
            logger.info(f"Final results saved: {full_path}")
        except Exception as exc:
            logger.error(f"Failed to save final results to {full_path}: {exc}")
            raise
    
    def _sync_save(self, output_path: Path):
        """Synchronous save helper."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_jobs, f, indent=2, ensure_ascii=False)
    
    async def process_from_file_async(
        self,
        scraped_data_file: str,
        output_filename: str = "processed_jobs.json"
    ) -> Tuple[List[Dict], ProcessingStats]:
        """Convenience method for async processing from file."""
        scraped_jobs = self.load_scraped_jobs(scraped_data_file)
        return await self.process_jobs_async(scraped_jobs, output_filename)


# Example usage with different concurrency levels
if __name__ == "__main__":
    import time
    load_dotenv()
    
    input_file = "/Users/jdtarriela/Documents/git/RoleRadar/src/job_data/capitalone_jobs_1759042384.json"
    
    # Configuration for different performance levels
    configs = {
        "conservative": {
            "input_file": input_file,
            "output_filename": "processed_jobs_conservative.json",
            "max_concurrent_requests": 3,  # Safe for most APIs
            "batch_size": 15,
            "output_dir": "processed_jobs"
        },
        "balanced": {
            "input_file": input_file, 
            "output_filename": "processed_jobs_balanced.json",
            "max_concurrent_requests": 5,  # Good balance
            "batch_size": 20,
            "output_dir": "processed_jobs"
        },
        "aggressive": {
            "input_file": input_file,
            "output_filename": "processed_jobs_aggressive.json", 
            "max_concurrent_requests": 25,  # Higher risk of rate limits
            "batch_size": 50,
            "output_dir": "processed_jobs"
        }
    }
    
    # Choose your performance level
    config = configs["aggressive"]  # Start with balanced
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def main():
        # Create async processor
        processor = AsyncJobProcessor(
            max_concurrent_requests=config["max_concurrent_requests"],
            batch_size=config["batch_size"],
            output_dir=config["output_dir"]
        )
        
        try:
            start_time = time.time()
            processed_jobs, stats = await processor.process_from_file_async(
                config["input_file"],
                config["output_filename"]
            )
            
            total_time = time.time() - start_time
            throughput = len(processed_jobs) / total_time if total_time > 0 else 0
            
            print(f"\n🎉 Processing Results:")
            print(f"   Total processed: {stats.successful_parses}/{stats.total_jobs}")
            print(f"   Success rate: {stats.success_rate:.1f}%")
            print(f"   Total time: {total_time:.1f} seconds")
            print(f"   Throughput: {throughput:.1f} jobs/second")
            print(f"   Estimated cost savings: ~{total_time/1.5:.1f}s vs sequential")
            
        except Exception as e:
            logger.error(f"Async processing failed: {e}")
    
    # Run the async processor
    asyncio.run(main())