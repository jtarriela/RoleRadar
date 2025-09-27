import os
import json
import asyncio
import time
import re
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs
import google.generativeai as genai
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# =================== CONFIGURATION ===================
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBRLSOoung9KHxvWcxbrEEhgCLNBio3bF4")  # Move to env!
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# =================== HELPER FUNCTIONS ===================
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
    text = text.replace('\\', '\\\\')
    text = text.replace('"', '\\"')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    return text

# =================== METRICS ===================
@dataclass
class PerformanceMetrics:
    start_time: float
    jobs_discovered: int = 0
    job_pages_crawled: int = 0
    jobs_extracted: int = 0
    llm_calls: int = 0
    llm_errors: int = 0
    
    def print_stats(self):
        elapsed = time.time() - self.start_time
        print(f"\n📊 Performance Stats:")
        print(f"  ⏱️  Time elapsed: {elapsed:.2f}s")
        print(f"  🔍 Jobs discovered: {self.jobs_discovered}")
        print(f"  📄 Job pages crawled: {self.job_pages_crawled}")
        print(f"  💼 Jobs extracted: {self.jobs_extracted}")
        print(f"  🤖 LLM calls made: {self.llm_calls}")
        print(f"  ❌ LLM errors: {self.llm_errors}")
        print(f"  ⚡ Pages/second: {self.job_pages_crawled/elapsed:.2f}")
        print(f"  ⚡ Efficiency: {self.jobs_extracted}/{self.job_pages_crawled} = {self.jobs_extracted/max(self.job_pages_crawled,1):.2%}")

# =================== BOFA JOB CRAWLER ===================
class BofAJobCrawler:
    """Specialized crawler for Bank of America careers site"""
    
    def __init__(self):
        self.job_urls = set()
        
    async def get_view_all_url(self, start_url: str) -> str:
        """
        Find the "View all" link on the page or construct it
        """
        print(f"\n🔍 Looking for 'View all' link on {start_url}")
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=3.0,
            js_code="""
            // Look for "View all" link - specifically the one with high row count
            const allLinks = document.querySelectorAll('a[href*="rows="]');
            let maxRows = 0;
            let bestLink = null;
            
            allLinks.forEach(link => {
                const href = link.href;
                const rowsMatch = href.match(/rows=(\\d+)/);
                if (rowsMatch && href.includes('getAllJobs')) {
                    const rows = parseInt(rowsMatch[1]);
                    if (rows > maxRows) {
                        maxRows = rows;
                        bestLink = href;
                    }
                }
            });
            
            if (bestLink) {
                console.log('Found View All with max rows:', bestLink);
            }
            """
        )
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(start_url, config)
            
            # Extract HTML
            if hasattr(result, '__aiter__'):
                async for r in result:
                    html = r.html if hasattr(r, 'html') else str(r)
                    break
            else:
                html = result.html if hasattr(result, 'html') else str(result)
            
            # Look for "View all" link patterns - prioritize ones with high row counts
            # and that are NOT student links
            all_matches = []
            
            # Find all potential View All links
            patterns = [
                r'href="([^"]*job-search[^"]*rows=(\d+)[^"]*getAllJobs[^"]*)"',
                r'href="([^"]*getAllJobs[^"]*rows=(\d+)[^"]*)"',
                r'<a[^>]*href="([^"]*rows=(\d+)[^"]*)"[^>]*>[^<]*View\s*[Aa]ll'
            ]
            
            for pattern in patterns:
                for match in re.finditer(pattern, html, re.IGNORECASE):
                    url = match.group(1)
                    rows = int(match.group(2))
                    
                    # Skip student links and low row counts
                    if 'students' not in url and rows > 10:
                        all_matches.append((url, rows))
            
            # Sort by row count and pick the highest
            if all_matches:
                all_matches.sort(key=lambda x: x[1], reverse=True)
                best_url, best_rows = all_matches[0]
                
                if not best_url.startswith('http'):
                    best_url = urljoin(start_url, best_url)
                
                # Clean up HTML entities
                best_url = best_url.replace('&amp;', '&')
                
                print(f"  ✅ Found 'View all' link with {best_rows} jobs: {best_url}")
                return best_url
            
            # If no good View All found, construct it directly
            # Try to extract current total from the page
            total_match = re.search(r'(\d+)\s*(?:jobs?|results?|positions?)\s*found', html, re.IGNORECASE)
            if total_match:
                total_jobs = int(total_match.group(1))
                print(f"  📊 Found {total_jobs} total jobs on page, constructing URL")
                return f"https://careers.bankofamerica.com/en-us/job-search?ref=search&rows={total_jobs + 100}&search=getAllJobs"
            
            # Default fallback - try a large number
            print("  ⚠️ No 'View all' link found, using fallback URL with rows=2000")
            return "https://careers.bankofamerica.com/en-us/job-search?ref=search&rows=2000&search=getAllJobs"
    
    def extract_job_urls(self, content: str) -> Set[str]:
        """Extract BofA job detail URLs from the page"""
        job_urls = set()
        
        # BofA job detail URL pattern
        pattern = r'href="(/en-us/job-detail/\d+/[^"]+)"'
        
        for match in re.finditer(pattern, content, re.IGNORECASE):
            relative_url = match.group(1)
            full_url = f"https://careers.bankofamerica.com{relative_url}"
            job_urls.add(full_url)
        
        # Also try full URL pattern
        full_pattern = r'https://careers\.bankofamerica\.com/en-us/job-detail/\d+/[^"\'<>\s]+'
        for match in re.finditer(full_pattern, content, re.IGNORECASE):
            job_urls.add(match.group(0))
        
        return job_urls
    
    async def crawl_all_jobs(self, start_url: str = None) -> Set[str]:
        """
        Main entry point - gets all BofA jobs
        """
        if start_url is None:
            start_url = "https://careers.bankofamerica.com/en-us/job-search?ref=search&start=0&rows=10&search=getAllJobs"
        
        # Step 1: Get the "View all" URL
        view_all_url = await self.get_view_all_url(start_url)
        
        # Step 2: Load the page with all jobs
        print(f"\n📄 Loading all jobs from: {view_all_url}")
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=5.0,  # Give time for all jobs to load
            page_timeout=60000,  # 60 seconds for large page
            js_code="""
            // Scroll to load any lazy-loaded content
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(resolve => setTimeout(resolve, 2000));
            window.scrollTo(0, 0);
            """
        )
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(view_all_url, config)
            
            # Extract HTML
            if hasattr(result, '__aiter__'):
                async for r in result:
                    html = r.html if hasattr(r, 'html') else str(r)
                    break
            else:
                html = result.html if hasattr(result, 'html') else str(result)
            
            # Extract all job URLs
            self.job_urls = self.extract_job_urls(html)
            
            print(f"  ✅ Found {len(self.job_urls)} job listings")
            
            # Save HTML for debugging if needed
            if len(self.job_urls) < 100:
                with open("bofa_debug.html", "w", encoding="utf-8") as f:
                    f.write(html[:100000])
                print("  💾 Saved debug HTML (found fewer than expected jobs)")
        
        return self.job_urls

# =================== LLM PROCESSOR ===================
class BofAJobProcessor:
    """Process BofA job pages with Gemini"""
    
    def __init__(self, batch_size: int = 2):
        self.batch_size = batch_size
    
    async def process_single_job(self, url: str, markdown: str, metrics: PerformanceMetrics) -> Dict:
        """Process a single job page"""
        try:
            cleaned_md = drop_hyperlink_lines(markdown, max_chars=2500)
            cleaned_md = safe_json_string(cleaned_md)
            
            prompt = f"""
            Analyze this Bank of America job posting and extract the details.
            
            URL: {url}
            Content: {cleaned_md}
            
            Return ONLY a JSON object:
            {{
              "job_title": "...",
              "location": "...",
              "department": "...",
              "job_type": "Full-time/Part-time/Contract",
              "summary": "...",
              "qualifications": ["..."],
              "responsibilities": ["..."],
              "salary_range": "...",
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
            data["url"] = url
            return data
            
        except Exception as e:
            metrics.llm_errors += 1
            print(f"    ⚠️ Error processing {url[:50]}: {str(e)[:100]}")
            return None
    
    async def batch_process(self, pages: List[Tuple[str, str]], metrics: PerformanceMetrics) -> List[Dict]:
        """Process multiple pages"""
        if not pages:
            return []
        
        print(f"\n🚀 Processing batch of {len(pages)} pages...")
        
        # Try batch processing first
        try:
            pages_data = []
            for url, markdown in pages:
                cleaned_md = drop_hyperlink_lines(markdown, max_chars=2000)
                cleaned_md = safe_json_string(cleaned_md)
                pages_data.append(f'{{"url": "{url}", "content": "{cleaned_md}"}}')
            
            pages_json_str = "[" + ",".join(pages_data) + "]"
            
            prompt = f"""
            Analyze these {len(pages)} Bank of America job postings.
            Extract details for each job.
            
            Pages: {pages_json_str}
            
            Return ONLY a JSON array:
            [
              {{
                "url": "...",
                "job_title": "...",
                "location": "...",
                "department": "...",
                "summary": "...",
                "qualifications": ["..."]
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
            
            for result in results:
                print(f"  ✅ Extracted: {result.get('job_title', 'Unknown')[:50]}")
            
            return results
            
        except Exception as e:
            print(f"  ⚠️ Batch failed: {str(e)[:100]}")
            metrics.llm_errors += 1
            
            # Fallback to individual processing
            print("  🔄 Processing individually...")
            results = []
            for url, markdown in pages:
                job = await self.process_single_job(url, markdown, metrics)
                if job:
                    results.append(job)
                    print(f"  ✅ Extracted: {job.get('job_title', 'Unknown')[:50]}")
                await asyncio.sleep(0.3)
            
            return results

# =================== PARALLEL CRAWLER ===================
class ParallelCrawler:
    """Crawl multiple pages concurrently"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def crawl_page(self, crawler, url: str, config) -> Tuple[str, str]:
        """Crawl a single page"""
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
    
    async def crawl_many(self, urls: List[str], metrics: PerformanceMetrics) -> List[Tuple[str, str]]:
        """Crawl multiple URLs in parallel"""
        total = len(urls)
        print(f"\n🌐 Crawling {total} job pages with {self.max_concurrent} concurrent connections")
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=1.5
        )
        
        async with AsyncWebCrawler() as crawler:
            crawled_pages = []
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

# =================== MAIN ===================
async def main():
    """Main orchestrator for BofA job scraping"""
    
    metrics = PerformanceMetrics(start_time=time.time())
    
    print("=" * 70)
    print("🏦 BANK OF AMERICA JOB SCRAPER")
    print("=" * 70)
    
    # Step 1: Discover all job URLs
    crawler = BofAJobCrawler()
    job_urls = await crawler.crawl_all_jobs()
    metrics.jobs_discovered = len(job_urls)
    
    if not job_urls:
        print("❌ No jobs found. Check the website structure.")
        return
    
    print(f"\n{'='*70}")
    print(f"📋 DISCOVERED {len(job_urls)} JOB LISTINGS")
    print(f"{'='*70}")
    
    # Save discovered URLs
    with open("bofa_job_urls.json", "w") as f:
        json.dump(list(job_urls), f, indent=2)
    print(f"💾 Saved URL list to bofa_job_urls.json")
    
    # Optional: Limit for testing
    # job_urls = list(job_urls)[:50]  # Uncomment to test with first 50 jobs
    
    # Step 2: Crawl job pages
    parallel_crawler = ParallelCrawler(max_concurrent=10)
    crawled_pages = await parallel_crawler.crawl_many(list(job_urls), metrics)
    
    # Step 3: Process with LLM
    processor = BofAJobProcessor(batch_size=2)
    all_jobs = []
    
    print(f"\n{'='*70}")
    print(f"🤖 PROCESSING WITH AI")
    print(f"{'='*70}")
    
    batch_size = 2
    total_batches = (len(crawled_pages) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(crawled_pages), batch_size), 1):
        batch = crawled_pages[i:i+batch_size]
        print(f"\n📦 Batch {batch_num}/{total_batches}")
        
        jobs = await processor.batch_process(batch, metrics)
        all_jobs.extend(jobs)
        metrics.jobs_extracted = len(all_jobs)
        
        # Progress update every 10 batches
        if batch_num % 10 == 0:
            print(f"\n  Progress: {metrics.jobs_extracted}/{metrics.jobs_discovered} jobs processed")
        
        if i + batch_size < len(crawled_pages):
            await asyncio.sleep(0.5)
    
    # Save results
    output_file = f"bofa_jobs_{len(all_jobs)}_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "total_jobs_found": len(all_jobs),
        "total_urls_discovered": len(job_urls),
        "job_pages_crawled": metrics.job_pages_crawled,
        "extraction_success_rate": f"{metrics.jobs_extracted}/{metrics.job_pages_crawled}",
        "llm_errors": metrics.llm_errors,
        "sample_jobs": [
            {
                "title": job.get("job_title", "Unknown"),
                "location": job.get("location", "Unknown"),
                "url": job.get("url", "")
            }
            for job in all_jobs[:10]  # First 10 as sample
        ]
    }
    
    with open("bofa_scrape_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print final stats
    print("\n" + "=" * 70)
    print(f"✅ SCRAPING COMPLETE!")
    print(f"💼 Successfully extracted {len(all_jobs)} jobs")
    print(f"📁 Results saved to {output_file}")
    print(f"📊 Summary saved to bofa_scrape_summary.json")
    metrics.print_stats()
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())