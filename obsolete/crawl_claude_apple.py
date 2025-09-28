import os
import json
import asyncio
import time
import re
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
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

def safe_json_string(text: str) -> str:
    """Escape text for safe inclusion in JSON"""
    # Remove or escape problematic characters
    text = text.replace('\\', '\\\\')  # Escape backslashes first
    text = text.replace('"', '\\"')    # Escape quotes
    text = text.replace('\n', ' ')     # Replace newlines with space
    text = text.replace('\r', ' ')     # Replace carriage returns
    text = text.replace('\t', ' ')     # Replace tabs
    # Remove any other control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    return text

# =================== PERFORMANCE METRICS ===================
@dataclass
class PerformanceMetrics:
    start_time: float
    search_pages_crawled: int = 0
    job_pages_crawled: int = 0
    jobs_extracted: int = 0
    llm_calls: int = 0
    llm_errors: int = 0
    
    def print_stats(self):
        elapsed = time.time() - self.start_time
        print(f"\n📊 Performance Stats:")
        print(f"  ⏱️  Time elapsed: {elapsed:.2f}s")
        print(f"  🔍 Search pages crawled: {self.search_pages_crawled}")
        print(f"  📄 Job pages crawled: {self.job_pages_crawled}")
        print(f"  💼 Jobs extracted: {self.jobs_extracted}")
        print(f"  🤖 LLM calls made: {self.llm_calls}")
        print(f"  ❌ LLM errors: {self.llm_errors}")
        print(f"  ⚡ Pages/second: {(self.search_pages_crawled + self.job_pages_crawled)/elapsed:.2f}")
        print(f"  ⚡ Efficiency: {self.jobs_extracted}/{self.job_pages_crawled} = {self.jobs_extracted/max(self.job_pages_crawled,1):.2%}")

# =================== SIMPLE PAGINATION CRAWLER ===================
class SimplePaginationCrawler:
    """Simple, reliable pagination crawler using URL construction"""
    
    def __init__(self):
        self.job_urls = set()
        
    def extract_job_urls(self, content: str, base_url: str) -> Set[str]:
        """Extract job detail URLs from search page"""
        job_urls = set()
        
        # Pattern for Apple job detail pages
        detail_patterns = [
            r'href="(/en-us/details/[^"]+)"',
            r'"url":\s*"([^"]*details[^"]*)"',
            r'data-href="([^"]*details[^"]*)"',
        ]
        
        for pattern in detail_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                relative_url = match.group(1)
                relative_url = relative_url.replace('\\/', '/').strip()
                if relative_url.startswith('http'):
                    full_url = relative_url
                else:
                    full_url = urljoin(base_url, relative_url)
                
                # Filter out non-job URLs
                if '/details/' in full_url and 'locationPicker' not in full_url:
                    job_urls.add(full_url)
        
        return job_urls
    
    async def crawl_with_pagination(self, start_url: str, max_pages: int = 50) -> Set[str]:
        """Crawl pages using URL construction with page parameter"""
        print(f"\n🔍 Starting pagination crawl from {start_url}")
        print(f"   Will crawl up to {max_pages} pages")
        
        all_job_urls = set()
        consecutive_empty = 0
        max_empty = 3
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=2.0
        )
        
        async with AsyncWebCrawler() as crawler:
            for page_num in range(1, max_pages + 1):
                # Construct URL with page number
                if page_num == 1:
                    page_url = start_url
                else:
                    # For Apple, use ?page=X pattern
                    parsed = urlparse(start_url)
                    query_params = parse_qs(parsed.query)
                    query_params['page'] = [str(page_num)]
                    new_query = urlencode(query_params, doseq=True)
                    page_url = urlunparse(parsed._replace(query=new_query))
                
                print(f"  📄 Page {page_num}/{max_pages}: {page_url}")
                
                try:
                    result = await crawler.arun(page_url, config)
                    
                    # Extract content
                    if hasattr(result, '__aiter__'):
                        async for r in result:
                            content = r.html if hasattr(r, 'html') else str(r)
                            break
                    else:
                        content = result.html if hasattr(result, 'html') else str(result)
                    
                    # Extract job URLs
                    new_urls = self.extract_job_urls(content, page_url)
                    
                    if new_urls:
                        before_count = len(all_job_urls)
                        all_job_urls.update(new_urls)
                        after_count = len(all_job_urls)
                        print(f"    ✅ Found {len(new_urls)} jobs ({after_count - before_count} new)")
                        consecutive_empty = 0
                    else:
                        consecutive_empty += 1
                        print(f"    ❌ No jobs found")
                        if consecutive_empty >= max_empty:
                            print(f"    🛑 Stopping after {max_empty} consecutive empty pages")
                            break
                    
                    await asyncio.sleep(0.5)  # Be polite
                    
                except Exception as e:
                    print(f"    ⚠️ Error on page {page_num}: {e}")
                    consecutive_empty += 1
                    if consecutive_empty >= max_empty:
                        break
        
        return all_job_urls

# =================== ROBUST BATCH LLM PROCESSOR ===================
class RobustBatchLLMProcessor:
    """Robust processor that handles errors gracefully"""
    
    def __init__(self, batch_size: int = 2):  # Even smaller for safety
        self.batch_size = batch_size
        
    async def process_single_page(self, url: str, markdown: str, metrics: PerformanceMetrics) -> Dict:
        """Process a single page - fallback method"""
        try:
            # Clean and truncate markdown
            cleaned_md = drop_hyperlink_lines(markdown, max_chars=2000)
            cleaned_md = safe_json_string(cleaned_md)
            
            prompt = f"""
            Analyze this job page. If it's a job description, extract the details.
            
            URL: {url}
            Content: {cleaned_md}
            
            Return ONLY a JSON object:
            {{
              "is_job_page": true/false,
              "job_title": "...",
              "location": "...",
              "summary": "...",
              "qualifications": ["..."],
              "pay_range": "...",
              "benefits": "..."
            }}
            """
            
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
                safety_settings={
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                }
            )
            
            metrics.llm_calls += 1
            
            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "")
            
            data = json.loads(text)
            if data.get("is_job_page"):
                data["url"] = url
                return data
                
        except Exception as e:
            metrics.llm_errors += 1
            print(f"    ⚠️ Single page processing error: {str(e)[:100]}")
        
        return None
        
    async def batch_analyze_and_extract(self, pages: List[Tuple[str, str]], metrics: PerformanceMetrics) -> List[Dict]:
        """
        Process pages - tries batch first, falls back to individual
        """
        if not pages:
            return []
        
        print(f"\n🚀 Processing {len(pages)} pages...")
        
        # Try batch processing first
        try:
            # Prepare safe JSON content
            pages_data = []
            for url, markdown in pages:
                cleaned_md = drop_hyperlink_lines(markdown, max_chars=2000)
                cleaned_md = safe_json_string(cleaned_md)
                pages_data.append(f'{{"url": "{url}", "content": "{cleaned_md}"}}')
            
            pages_json_str = "[" + ",".join(pages_data) + "]"
            
            prompt = f"""
            Analyze these {len(pages)} job pages. For each, determine if it's a job description.
            Extract job details only if it IS a job page.
            
            Pages: {pages_json_str}
            
            Return ONLY a JSON array with one object per page:
            [
              {{
                "url": "...",
                "is_job_page": true/false,
                "job_title": "...",
                "location": "...",
                "summary": "...",
                "qualifications": ["..."],
                "pay_range": "...",
                "benefits": "..."
              }}
            ]
            """
            
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
                safety_settings={
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                }
            )
            
            metrics.llm_calls += 1
            
            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "")
            
            results = json.loads(text)
            
            job_data_list = []
            for result in results:
                if result.get("is_job_page"):
                    job_data_list.append(result)
                    print(f"  ✅ Extracted: {result.get('job_title', 'Unknown')[:50]}")
            
            return job_data_list
            
        except Exception as e:
            print(f"  ⚠️ Batch failed ({str(e)[:100]}), trying individual processing...")
            metrics.llm_errors += 1
            
            # Fallback: process individually
            job_data_list = []
            for url, markdown in pages:
                job_data = await self.process_single_page(url, markdown, metrics)
                if job_data:
                    job_data_list.append(job_data)
                    print(f"  ✅ Extracted: {job_data.get('job_title', 'Unknown')[:50]}")
                await asyncio.sleep(0.3)  # Rate limit
            
            return job_data_list

# =================== PARALLEL CRAWLER ===================
class ParallelJobCrawler:
    """Crawls multiple job detail pages concurrently"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def crawl_page(self, crawler, url: str, config) -> Tuple[str, str]:
        """Crawl a single page with semaphore for concurrency control"""
        async with self.semaphore:
            try:
                result = await crawler.arun(url, config)
                
                if hasattr(result, '__aiter__'):
                    async for r in result:
                        if r.markdown and r.markdown.raw_markdown:
                            return (r.url, r.markdown.raw_markdown)
                elif result.markdown and result.markdown.raw_markdown:
                    return (result.url, result.markdown.raw_markdown)
                    
                return (url, "")
            except Exception as e:
                return (url, "")
    
    async def crawl_many(self, urls: List[str], config, metrics: PerformanceMetrics) -> List[Tuple[str, str]]:
        """Crawl multiple URLs in parallel"""
        total = len(urls)
        print(f"\n🌐 Crawling {total} job pages with {self.max_concurrent} concurrent connections")
        
        async with AsyncWebCrawler() as crawler:
            crawled_pages = []
            
            # Process in chunks
            chunk_size = self.max_concurrent * 2
            for i in range(0, total, chunk_size):
                chunk = list(urls)[i:i+chunk_size]
                print(f"  📦 Batch {i//chunk_size + 1}: Jobs {i+1}-{min(i+chunk_size, total)} of {total}")
                
                tasks = [self.crawl_page(crawler, url, config) for url in chunk]
                results = await asyncio.gather(*tasks)
                
                for url, content in results:
                    if content:
                        crawled_pages.append((url, content))
                        metrics.job_pages_crawled += 1
                
                if i + chunk_size < total:
                    await asyncio.sleep(0.5)
        
        print(f"  ✅ Successfully crawled {len(crawled_pages)}/{total} pages")
        return crawled_pages

# =================== MAIN ORCHESTRATOR ===================
async def main(max_search_pages: int = 10):
    """
    Main scraper with configurable page limit
    
    Args:
        max_search_pages: Maximum number of search result pages to crawl (default 50)
    """
    
    metrics = PerformanceMetrics(start_time=time.time())
    
    # Configuration
    START_URL = "https://jobs.apple.com/en-us/search"
    BATCH_SIZE = 2  # Small batch for reliability
    MAX_CONCURRENT_CRAWLS = 10  # Parallel crawling connections
    
    print("=" * 70)
    print("🚀 ROBUST JOB SCRAPER v5.0")
    print(f"   Target: {START_URL}")
    print(f"   Max pages to crawl: {max_search_pages}")
    print("=" * 70)
    
    # Step 1: Crawl search pages with pagination
    crawler = SimplePaginationCrawler()
    all_job_urls = await crawler.crawl_with_pagination(START_URL, max_search_pages)
    metrics.search_pages_crawled = max_search_pages  # Approximate
    
    if not all_job_urls:
        print("❌ No job URLs discovered.")
        return
    
    print(f"\n{'='*70}")
    print(f"📋 DISCOVERED {len(all_job_urls)} TOTAL JOB LISTINGS")
    print(f"{'='*70}")
    
    # Save discovered URLs
    with open("discovered_job_urls.json", "w") as f:
        json.dump(list(all_job_urls), f, indent=2)
    print(f"💾 Saved URL list to discovered_job_urls.json")
    
    # Step 2: Parallel crawl job pages
    job_crawler = ParallelJobCrawler(max_concurrent=MAX_CONCURRENT_CRAWLS)
    
    config = CrawlerRunConfig(
        verbose=False,
        wait_for="css:body",
        delay_before_return_html=1.0
    )
    
    crawled_pages = await job_crawler.crawl_many(all_job_urls, config, metrics)
    
    # Step 3: Process with LLM
    processor = RobustBatchLLMProcessor(batch_size=BATCH_SIZE)
    all_jobs = []
    
    print(f"\n{'='*70}")
    print(f"🤖 PROCESSING WITH LLM")
    print(f"{'='*70}")
    
    # Process in batches
    total_batches = (len(crawled_pages) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num, i in enumerate(range(0, len(crawled_pages), BATCH_SIZE), 1):
        batch = crawled_pages[i:i+BATCH_SIZE]
        print(f"\n📦 Batch {batch_num}/{total_batches}")
        
        jobs = await processor.batch_analyze_and_extract(batch, metrics)
        all_jobs.extend(jobs)
        metrics.jobs_extracted = len(all_jobs)
        
        if i + BATCH_SIZE < len(crawled_pages):
            await asyncio.sleep(0.5)
    
    # Save results
    output_file = f"apple_jobs_{len(all_jobs)}_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "total_jobs_found": len(all_jobs),
        "total_urls_discovered": len(all_job_urls),
        "search_pages_crawled": metrics.search_pages_crawled,
        "job_pages_crawled": metrics.job_pages_crawled,
        "llm_errors": metrics.llm_errors,
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
    import sys
    
    # Check for command line argument for max pages
    max_pages = 5  # Default
    if len(sys.argv) > 1:
        try:
            max_pages = int(sys.argv[1])
            print(f"Using custom max pages: {max_pages}")
        except ValueError:
            print(f"Invalid page count: {sys.argv[1]}, using default: {max_pages}")
    
    asyncio.run(main(max_pages))
