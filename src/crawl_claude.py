import os
import json
import asyncio
import time
import re
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import google.generativeai as genai
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# =================== CONFIGURATION ===================
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBRLSOoung9KHxvWcxbrEEhgCLNBio3bF4")  # Move to env!
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# =================== HYPERLINK STRIPPER ===================
URL_LINE_RE = re.compile(r"""
(
    \[[^\]]+\]\([^)]+\)                # [text](url)
  | <https?://[^>\s]+>                 # <https://...>
  | https?://\S+                       # raw http(s)
  | \b(?:www\.)?[a-z0-9][\w.-]+\.(?:com|org|net|io|co|edu|gov)(?:/\S*)?
)
""", re.IGNORECASE | re.VERBOSE)

def drop_hyperlink_lines(md: str, max_chars: int | None = None) -> str:
    """Remove any entire line that contains a hyperlink; optionally cap to max_chars."""
    out = []
    for line in md.splitlines():
        if URL_LINE_RE.search(line):
            continue
        out.append(line)
    cleaned = "\n".join(out)
    return cleaned[:max_chars] if isinstance(max_chars, int) and max_chars > 0 else cleaned

# =================== PERFORMANCE METRICS ===================
@dataclass
class PerformanceMetrics:
    start_time: float
    search_pages_crawled: int = 0
    job_pages_crawled: int = 0
    jobs_extracted: int = 0
    llm_calls: int = 0
    
    def print_stats(self):
        elapsed = time.time() - self.start_time
        print(f"\n📊 Performance Stats:")
        print(f"  ⏱️  Time elapsed: {elapsed:.2f}s")
        print(f"  🔍 Search pages crawled: {self.search_pages_crawled}")
        print(f"  📄 Job pages crawled: {self.job_pages_crawled}")
        print(f"  💼 Jobs extracted: {self.jobs_extracted}")
        print(f"  🤖 LLM calls made: {self.llm_calls}")
        print(f"  ⚡ Pages/second: {(self.search_pages_crawled + self.job_pages_crawled)/elapsed:.2f}")
        print(f"  ⚡ Efficiency: {self.jobs_extracted}/{self.job_pages_crawled} = {self.jobs_extracted/max(self.job_pages_crawled,1):.2%}")

# =================== PAGINATION CRAWLER ===================
class PaginationCrawler:
    """Crawls through paginated search results and collects all job URLs"""
    
    def __init__(self):
        self.visited_pages = set()
        self.job_urls = set()
        
    def extract_job_urls(self, html_content: str, base_url: str) -> Set[str]:
        """Extract job detail URLs from search page"""
        job_urls = set()
        
        # Pattern for Apple job detail pages
        # Looking for URLs like: /en-us/details/[job-id]/[job-title]
        detail_pattern = re.compile(r'href="(/en-us/details/[^"]+)"')
        
        for match in detail_pattern.finditer(html_content):
            relative_url = match.group(1)
            full_url = urljoin(base_url, relative_url)
            job_urls.add(full_url)
        
        return job_urls
    
    def extract_next_page_url(self, html_content: str, base_url: str) -> str:
        """Extract the 'next page' URL from search results"""
        # Apple uses pagination with patterns like ?page=2, ?page=3, etc.
        # Or sometimes uses a "Show more" button
        
        # Look for next page link patterns
        patterns = [
            r'<a[^>]*href="([^"]*\?page=\d+[^"]*)"[^>]*>.*?(?:Next|Show\s*more|Load\s*more)',
            r'href="([^"]*\?page=\d+[^"]*)"[^>]*class="[^"]*next[^"]*"',
            r'data-href="([^"]*\?page=\d+[^"]*)"',
            # Apple specific: look for pagination in their search API calls
            r'"next":\s*"([^"]+)"',
            r'<a[^>]*class="[^"]*pagination[^"]*"[^>]*href="([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                next_url = match.group(1)
                if not next_url.startswith('http'):
                    next_url = urljoin(base_url, next_url)
                return next_url
        
        return None
    
    async def crawl_all_search_pages(self, start_url: str, max_pages: int = 50) -> Set[str]:
        """
        Crawl through all search result pages following pagination.
        Returns all discovered job detail URLs.
        """
        print(f"\n🔍 Starting pagination crawl from {start_url}")
        print(f"   Will crawl up to {max_pages} search pages")
        
        current_url = start_url
        pages_crawled = 0
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=2.0,  # Wait for dynamic content
            js_code="""
            // Try to click any "Show More" buttons if they exist
            const showMoreButtons = document.querySelectorAll('button:contains("Show More"), button:contains("Load More"), a:contains("Next")');
            showMoreButtons.forEach(btn => btn.click());
            """
        )
        
        async with AsyncWebCrawler() as crawler:
            while current_url and pages_crawled < max_pages:
                if current_url in self.visited_pages:
                    print(f"  ⚠️ Already visited {current_url}, stopping to avoid loop")
                    break
                
                self.visited_pages.add(current_url)
                pages_crawled += 1
                
                print(f"  📄 Crawling search page {pages_crawled}: {current_url}")
                
                try:
                    result = await crawler.arun(current_url, config)
                    
                    # Handle the result (could be single or generator)
                    if hasattr(result, '__aiter__'):
                        async for r in result:
                            html = r.html if hasattr(r, 'html') else str(r)
                            break
                    else:
                        html = result.html if hasattr(result, 'html') else str(result)
                    
                    # Extract job URLs from this page
                    new_job_urls = self.extract_job_urls(html, current_url)
                    before_count = len(self.job_urls)
                    self.job_urls.update(new_job_urls)
                    after_count = len(self.job_urls)
                    
                    print(f"    ✅ Found {len(new_job_urls)} job links ({after_count - before_count} new)")
                    
                    # Try to find next page
                    next_url = self.extract_next_page_url(html, current_url)
                    
                    if next_url and next_url != current_url:
                        print(f"    ➡️ Found next page: {next_url}")
                        current_url = next_url
                        await asyncio.sleep(0.5)  # Be polite between pages
                    else:
                        print(f"    🏁 No more pages found. Reached end of pagination.")
                        break
                        
                except Exception as e:
                    print(f"  ⚠️ Error crawling {current_url}: {e}")
                    break
        
        print(f"\n✅ Pagination complete! Crawled {pages_crawled} search pages")
        print(f"   Total unique job URLs discovered: {len(self.job_urls)}")
        
        return self.job_urls

# =================== BATCH LLM PROCESSOR ===================
class BatchLLMProcessor:
    """Processes multiple pages in a single LLM call for maximum efficiency"""
    
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        
    async def batch_analyze_and_extract(self, pages: List[Tuple[str, str]], metrics: PerformanceMetrics) -> List[Dict]:
        """
        Single LLM call to analyze multiple pages at once.
        Returns extracted job data for all pages in one go.
        """
        print(f"\n🚀 Batch processing {len(pages)} pages in a single LLM call...")
        
        # Build a consolidated prompt with all pages
        pages_json = []
        for url, markdown in pages:
            cleaned_md = drop_hyperlink_lines(markdown, max_chars=5000)  # Limit each page
            pages_json.append({
                "url": url,
                "content": cleaned_md
            })
        
        prompt = f"""
        You are an expert job data extractor. Analyze the following {len(pages)} web pages.
        For each page, determine if it's a single job description page (not a listing of multiple jobs).
        
        If it IS a job page, extract the job details. If NOT, mark it as not_job_page.
        
        Return a JSON array with one object per page:
        [
          {{
            "url": "...",
            "is_job_page": true/false,
            "job_data": {{  // Only include if is_job_page is true
              "job_title": "...",
              "location": "...",
              "summary": "...",
              "qualifications": [...],
              "pay_range": "...",
              "benefits": "..."
            }}
          }},
          ...
        ]
        
        Pages to analyze:
        {json.dumps(pages_json, indent=2)}
        
        IMPORTANT: Return ONLY the JSON array, no other text.
        """
        
        try:
            response = await model.generate_content_async(prompt)
            metrics.llm_calls += 1
            
            # Parse response
            cleaned_text = response.text.strip()
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
            
            results = json.loads(cleaned_text)
            
            # Extract job data from results
            job_data_list = []
            for result in results:
                if result.get("is_job_page") and result.get("job_data"):
                    job_data = result["job_data"]
                    job_data["url"] = result["url"]
                    job_data_list.append(job_data)
                    print(f"  ✅ Extracted: {job_data.get('job_title', 'Unknown')} at {result['url']}")
                else:
                    print(f"  ❌ Not a job page: {result['url']}")
            
            return job_data_list
            
        except Exception as e:
            print(f"  ⚠️ Batch LLM processing failed: {e}")
            return []

# =================== PARALLEL CRAWLER ===================
class ParallelJobCrawler:
    """Crawls multiple job detail pages concurrently for maximum throughput"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def crawl_page(self, crawler, url: str, config) -> Tuple[str, str]:
        """Crawl a single page with semaphore for concurrency control"""
        async with self.semaphore:
            try:
                result = await crawler.arun(url, config)
                
                # Handle both single result and async generator
                if hasattr(result, '__aiter__'):
                    async for r in result:
                        if r.markdown and r.markdown.raw_markdown:
                            return (r.url, r.markdown.raw_markdown)
                elif result.markdown and result.markdown.raw_markdown:
                    return (result.url, result.markdown.raw_markdown)
                    
                return (url, "")
            except Exception as e:
                print(f"  ⚠️ Failed to crawl {url}: {e}")
                return (url, "")
    
    async def crawl_many(self, urls: List[str], config, metrics: PerformanceMetrics) -> List[Tuple[str, str]]:
        """Crawl multiple URLs in parallel with progress tracking"""
        total = len(urls)
        print(f"\n🌐 Starting parallel crawl of {total} job pages")
        print(f"   Using {self.max_concurrent} concurrent connections")
        
        async with AsyncWebCrawler() as crawler:
            crawled_pages = []
            
            # Process in chunks to show progress
            chunk_size = self.max_concurrent * 2
            for i in range(0, total, chunk_size):
                chunk = list(urls)[i:i+chunk_size]
                print(f"\n  📦 Processing batch {i//chunk_size + 1}/{(total + chunk_size - 1)//chunk_size}")
                print(f"     Jobs {i+1}-{min(i+chunk_size, total)} of {total}")
                
                tasks = [self.crawl_page(crawler, url, config) for url in chunk]
                results = await asyncio.gather(*tasks)
                
                # Filter out empty results and update metrics
                for url, content in results:
                    if content:
                        crawled_pages.append((url, content))
                        metrics.job_pages_crawled += 1
                        print(f"    ✅ Crawled: {url.split('/')[-1][:50]}...")
                
                # Small delay between chunks
                if i + chunk_size < total:
                    await asyncio.sleep(0.5)
        
        return crawled_pages

# =================== MAIN ORCHESTRATOR ===================
async def main():
    """
    High-performance job scraping with:
    1. Full pagination crawling to discover ALL job URLs
    2. Concurrent crawling of all job detail pages
    3. Batch LLM processing for maximum efficiency
    """
    
    metrics = PerformanceMetrics(start_time=time.time())
    
    # Configuration
    START_URL = "https://jobs.apple.com/en-us/search"
    MAX_SEARCH_PAGES = 300  # Maximum search result pages to crawl
    BATCH_SIZE = 20  # How many pages to process in single LLM call
    MAX_CONCURRENT_CRAWLS = 15  # Parallel crawling connections
    
    print("=" * 70)
    print("🚀 HIGH-PERFORMANCE JOB SCRAPER WITH PAGINATION v3.0")
    print("=" * 70)
    
    # Step 1: Crawl all search pages and discover job URLs
    pagination_crawler = PaginationCrawler()
    all_job_urls = await pagination_crawler.crawl_all_search_pages(START_URL, MAX_SEARCH_PAGES)
    metrics.search_pages_crawled = len(pagination_crawler.visited_pages)
    
    if not all_job_urls:
        print("❌ No job URLs discovered. Exiting.")
        return
    
    print(f"\n{'='*70}")
    print(f"📋 DISCOVERED {len(all_job_urls)} TOTAL JOB LISTINGS")
    print(f"{'='*70}")
    
    # Optional: Save the list of discovered URLs for debugging
    with open("discovered_job_urls.json", "w") as f:
        json.dump(list(all_job_urls), f, indent=2)
    print(f"💾 Saved URL list to discovered_job_urls.json")
    
    # Step 2: Parallel crawl all job detail pages
    job_crawler = ParallelJobCrawler(max_concurrent=MAX_CONCURRENT_CRAWLS)
    
    # Simple config for direct page fetching
    simple_config = CrawlerRunConfig(
        verbose=False,
        wait_for="css:body",
        delay_before_return_html=1.0
    )
    
    crawled_pages = await job_crawler.crawl_many(all_job_urls, simple_config, metrics)
    
    print(f"\n✅ Successfully crawled {len(crawled_pages)}/{len(all_job_urls)} job pages")
    
    # Step 3: Batch process with LLM
    processor = BatchLLMProcessor(batch_size=BATCH_SIZE)
    all_jobs = []
    
    print(f"\n{'='*70}")
    print(f"🤖 STARTING LLM BATCH PROCESSING")
    print(f"{'='*70}")
    
    # Process in batches
    total_batches = (len(crawled_pages) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num, i in enumerate(range(0, len(crawled_pages), BATCH_SIZE), 1):
        batch = crawled_pages[i:i+BATCH_SIZE]
        print(f"\n📦 Processing batch {batch_num}/{total_batches}")
        
        jobs = await processor.batch_analyze_and_extract(batch, metrics)
        all_jobs.extend(jobs)
        metrics.jobs_extracted = len(all_jobs)
        
        # Small delay between batches to avoid rate limits
        if i + BATCH_SIZE < len(crawled_pages):
            await asyncio.sleep(0.5)
    
    # Save results
    output_file = "apple_jobs_complete.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "total_jobs_found": len(all_jobs),
        "search_pages_crawled": metrics.search_pages_crawled,
        "job_pages_crawled": metrics.job_pages_crawled,
        "extraction_rate": f"{metrics.jobs_extracted}/{metrics.job_pages_crawled}",
        "job_titles": [job.get("job_title", "Unknown") for job in all_jobs]
    }
    
    with open("scrape_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print final stats
    print("\n" + "=" * 70)
    print(f"✅ SCRAPING COMPLETE!")
    print(f"💼 Successfully extracted {len(all_jobs)} jobs from {len(all_job_urls)} URLs")
    print(f"📁 Results saved to {output_file}")
    print(f"📊 Summary saved to scrape_summary.json")
    metrics.print_stats()
    print("=" * 70)

# =================== ENTRY POINT ===================
if __name__ == "__main__":
    # For even more speed on multi-core systems, you could use:
    # import uvloop
    # uvloop.install()  # pip install uvloop (Linux/Mac only)
    asyncio.run(main())