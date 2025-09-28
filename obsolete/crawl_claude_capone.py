import os
import json
import asyncio
import time
import re
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urljoin

import google.generativeai as genai
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# =================== CONFIGURATION ===================
# Security: read API key from environment only (no hardcoded default)
# =================== CONFIGURATION ===================
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBRLSOoung9KHxvWcxbrEEhgCLNBio3bF4")  # Move to env!
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')


@dataclass
class ScrapingConfig:
    max_pages: int = 50
    max_concurrent: int = 8
    request_delay_range: Tuple[float, float] = (1.0, 3.0)
    retry_attempts: int = 3
    user_agents: List[str] = None
    
    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            ]

# =================== ENHANCED METRICS ===================
@dataclass
class EnhancedMetrics:
    start_time: float
    pages_discovered: int = 0
    pages_attempted: int = 0
    pages_successful: int = 0
    jobs_extracted: int = 0
    anti_bot_detections: int = 0
    fallback_activations: int = 0
    
    def success_rate(self) -> float:
        return self.pages_successful / max(self.pages_attempted, 1)
    
    def print_detailed_stats(self):
        elapsed = time.time() - self.start_time
        print(f"\n📊 Enhanced Performance Analysis:")
        print(f"  ⏱️  Total runtime: {elapsed:.2f}s")
        print(f"  🔍 Pages discovered: {self.pages_discovered}")
        print(f"  📄 Pages attempted: {self.pages_attempted}")
        print(f"  ✅ Pages successful: {self.pages_successful} ({self.success_rate():.1%})")
        print(f"  💼 Jobs extracted: {self.jobs_extracted}")
        print(f"  🛡️  Anti-bot hits: {self.anti_bot_detections}")
        print(f"  🔄 Fallback activations: {self.fallback_activations}")
        print(f"  ⚡ Efficiency: {self.jobs_extracted/max(elapsed,1):.2f} jobs/second")

# =================== ENHANCED CAPITAL ONE CRAWLER ===================
class EnhancedCapitalOneCrawler:
    """
    Production-grade Capital One job crawler with multiple discovery strategies:
    1. Direct URL construction (fastest, most reliable)
    2. Sitemap parsing (backup)
    3. Smart pagination clicking (last resort)
    """
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.discovered_urls: Set[str] = set()
        self.metrics = EnhancedMetrics(start_time=time.time())
        
    def _get_random_headers(self) -> Dict[str, str]:
        """Anti-detection: randomize headers"""
        return {
            'User-Agent': random.choice(self.config.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def _random_delay(self):
        """Anti-detection: random delays"""
        delay = random.uniform(*self.config.request_delay_range)
        await asyncio.sleep(delay)
    
    def _extract_job_urls_from_html(self, html: str, base_url: str) -> Set[str]:
        """Enhanced URL extraction with multiple patterns"""
        urls = set()
        
        # Pattern 1: Direct job links
        job_patterns = [
            r'href=["\']([^"\']*?/job/[^"\']*?)["\']',
            r'href=["\']([^"\']*?/job-detail/[^"\']*?)["\']',
            r'data-job-url=["\']([^"\']*?)["\']',
            r'<a[^>]+href=["\']([^"\']*?85\d{8,}[^"\']*?)["\']',  # Job ID pattern
        ]
        
        for pattern in job_patterns:
            matches = re.finditer(pattern, html, re.IGNORECASE)
            for match in matches:
                url = match.group(1)
                if url.startswith('/'):
                    url = urljoin(base_url, url)
                if 'capitalonecareers.com' in url and ('/job/' in url or '/job-detail/' in url):
                    urls.add(url)
        
        return urls
    
    async def strategy_1_direct_construction(self) -> Set[str]:
        """
        Strategy 1: Direct URL construction - most reliable approach
        Construct search result URLs directly instead of clicking through
        """
        print("🎯 Strategy 1: Direct URL Construction")
        base_search_url = "https://www.capitalonecareers.com/search-jobs/results"
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=2.0,
            page_timeout=30000,
            browser_type="chromium",
        )
        
        async with AsyncWebCrawler() as crawler:
            for page in range(1, min(self.config.max_pages + 1, 51)):  # Cap at 50 pages
                try:
                    # Different URL patterns Capital One might use
                    url_variants = [
                        f"{base_search_url}?p={page}",
                        f"{base_search_url}&p={page}",
                        f"{base_search_url}?page={page}",
                        f"https://www.capitalonecareers.com/search-jobs/?page={page}",
                    ]
                    
                    for url in url_variants:
                        self.metrics.pages_attempted += 1
                        
                        try:
                            result = await crawler.arun(url, config)
                            if hasattr(result, '__aiter__'):
                                async for r in result:
                                    html = getattr(r, 'html', '')
                                    break
                            else:
                                html = getattr(result, 'html', '')
                            
                            if html and len(html) > 5000:  # Meaningful content
                                urls = self._extract_job_urls_from_html(html, url)
                                if urls:
                                    before = len(self.discovered_urls)
                                    self.discovered_urls.update(urls)
                                    new_jobs = len(self.discovered_urls) - before
                                    print(f"  📄 Page {page}: +{new_jobs} jobs → {len(self.discovered_urls)} total")
                                    self.metrics.pages_successful += 1
                                    break  # Success with this URL variant
                                    
                        except Exception as e:
                            print(f"  ⚠️  Page {page} variant failed: {str(e)[:100]}")
                            continue
                    
                    await self._random_delay()
                    
                    # Stop if no new jobs found in last few pages
                    if page > 5 and len(self.discovered_urls) < page * 3:
                        print(f"  🛑 Stopping at page {page} - low job density detected")
                        break
                        
                except Exception as e:
                    print(f"  ❌ Page {page} completely failed: {str(e)[:100]}")
                    continue
        
        self.metrics.pages_discovered = len(self.discovered_urls)
        return self.discovered_urls
    
    async def strategy_2_sitemap_parsing(self) -> Set[str]:
        """
        Strategy 2: Parse sitemaps and robots.txt for job URLs
        """
        print("🗺️  Strategy 2: Sitemap Discovery")
        sitemap_urls = [
            "https://www.capitalonecareers.com/sitemap.xml",
            "https://www.capitalonecareers.com/sitemap_index.xml",
            "https://www.capitalonecareers.com/robots.txt",
        ]
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=1.0,
            page_timeout=20000,
        )
        
        async with AsyncWebCrawler() as crawler:
            for sitemap_url in sitemap_urls:
                try:
                    result = await crawler.arun(sitemap_url, config)
                    content = getattr(result, 'html', '') if hasattr(result, 'html') else str(result)
                    
                    # Extract URLs from XML sitemaps
                    url_matches = re.findall(r'<loc>(.*?)</loc>', content)
                    job_urls = {url for url in url_matches if '/job' in url}
                    
                    if job_urls:
                        before = len(self.discovered_urls)
                        self.discovered_urls.update(job_urls)
                        new_jobs = len(self.discovered_urls) - before
                        print(f"  🗺️  Sitemap: +{new_jobs} jobs from {sitemap_url}")
                        
                except Exception as e:
                    print(f"  ⚠️  Sitemap {sitemap_url} failed: {str(e)[:100]}")
        
        return self.discovered_urls
    
    async def strategy_3_smart_pagination(self) -> Set[str]:
        """
        Strategy 3: Enhanced pagination with better anti-detection
        """
        print("🔄 Strategy 3: Smart Pagination (Enhanced)")
        self.metrics.fallback_activations += 1
        
        # More sophisticated JavaScript with better timing
        js_code = """
        (async () => {
            const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
            const randomDelay = () => sleep(500 + Math.random() * 1500);
            
            const collectJobLinks = () => {
                const links = new Set();
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.getAttribute('href') || '';
                    if (href.includes('/job/') || href.includes('/job-detail/')) {
                        links.add(href);
                    }
                });
                return Array.from(links);
            };
            
            let allLinks = new Set(collectJobLinks());
            let currentPage = 1;
            const maxPages = 20;
            
            while (currentPage < maxPages) {
                await randomDelay();
                
                // Multiple strategies to find next button
                const nextSelectors = [
                    'a.next[href]',
                    'a[aria-label*="next"]',
                    'a[aria-label*="Next"]',
                    'button[aria-label*="next"]',
                    'a:contains("Next")',
                    'a[href*="p=' + (currentPage + 1) + '"]'
                ];
                
                let nextButton = null;
                for (const selector of nextSelectors) {
                    nextButton = document.querySelector(selector);
                    if (nextButton && nextButton.offsetParent !== null) break;
                }
                
                if (!nextButton) break;
                
                // Scroll to button naturally
                nextButton.scrollIntoView({ behavior: 'smooth', block: 'center' });
                await sleep(500 + Math.random() * 500);
                
                // Click with human-like behavior
                const rect = nextButton.getBoundingClientRect();
                const x = rect.left + rect.width * (0.3 + Math.random() * 0.4);
                const y = rect.top + rect.height * (0.3 + Math.random() * 0.4);
                
                nextButton.click();
                
                // Wait for content to load
                await sleep(2000 + Math.random() * 1000);
                
                const newLinks = collectJobLinks();
                const beforeSize = allLinks.size;
                newLinks.forEach(link => allLinks.add(link));
                
                if (allLinks.size === beforeSize) {
                    break; // No new content
                }
                
                currentPage++;
            }
            
            // Store results
            const resultsDiv = document.createElement('div');
            resultsDiv.id = '__job_links_found__';
            resultsDiv.style.display = 'none';
            resultsDiv.textContent = Array.from(allLinks).join('\\n');
            document.body.appendChild(resultsDiv);
        })();
        """
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=3.0,
            page_timeout=120000,  # Longer timeout for JS execution
            js_code=js_code,
        )
        
        start_url = "https://www.capitalonecareers.com/search-jobs/"
        
        async with AsyncWebCrawler() as crawler:
            try:
                result = await crawler.arun(start_url, config)
                html = getattr(result, 'html', '') if hasattr(result, 'html') else str(result)
                
                # Extract from JS results
                js_match = re.search(r'<div id="__job_links_found__"[^>]*>(.*?)</div>', html, re.DOTALL)
                if js_match:
                    js_links = {line.strip() for line in js_match.group(1).split('\n') if line.strip()}
                    before = len(self.discovered_urls)
                    self.discovered_urls.update(self._normalize_urls(js_links, start_url))
                    new_jobs = len(self.discovered_urls) - before
                    print(f"  🔄 Pagination: +{new_jobs} jobs via JavaScript")
                
                # Also extract from final HTML state
                html_urls = self._extract_job_urls_from_html(html, start_url)
                before = len(self.discovered_urls)
                self.discovered_urls.update(html_urls)
                new_jobs = len(self.discovered_urls) - before
                if new_jobs > 0:
                    print(f"  🔄 Pagination: +{new_jobs} additional jobs from HTML")
                    
            except Exception as e:
                print(f"  ❌ Smart pagination failed: {str(e)[:100]}")
                self.metrics.anti_bot_detections += 1
        
        return self.discovered_urls
    
    def _normalize_urls(self, urls: Set[str], base_url: str) -> Set[str]:
        """Normalize and validate URLs"""
        normalized = set()
        for url in urls:
            if url.startswith('/'):
                url = urljoin(base_url, url)
            if 'capitalonecareers.com' in url and ('/job/' in url or '/job-detail/' in url):
                normalized.add(url.replace('&amp;', '&'))
        return normalized
    
    async def discover_all_jobs(self) -> Set[str]:
        """
        Multi-strategy job discovery with fallbacks
        """
        print("🚀 Starting multi-strategy job discovery...")
        
        # Strategy 1: Direct construction (most reliable)
        try:
            await self.strategy_1_direct_construction()
        except Exception as e:
            print(f"❌ Strategy 1 failed: {e}")
        
        # Strategy 2: Sitemap parsing (if we need more)
        if len(self.discovered_urls) < 100:
            try:
                await self.strategy_2_sitemap_parsing()
            except Exception as e:
                print(f"❌ Strategy 2 failed: {e}")
        
        # Strategy 3: Smart pagination (last resort)
        if len(self.discovered_urls) < 50:
            try:
                await self.strategy_3_smart_pagination()
            except Exception as e:
                print(f"❌ Strategy 3 failed: {e}")
        
        print(f"✅ Discovery complete: {len(self.discovered_urls)} total jobs found")
        return self.discovered_urls

# =================== ENHANCED PROCESSOR ===================
class EnhancedJobProcessor:
    """Production-grade job content processor with better error handling"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async def process_job_batch(self, job_urls: List[str], metrics: EnhancedMetrics) -> List[Dict[str, Any]]:
        """Process multiple job URLs concurrently"""
        print(f"🤖 Processing {len(job_urls)} jobs with AI extraction...")
        
        async def process_single_job(url: str) -> Dict[str, Any] | None:
            async with self.semaphore:
                for attempt in range(self.config.retry_attempts):
                    try:
                        await asyncio.sleep(random.uniform(0.1, 0.5))  # Anti-detection
                        
                        config = CrawlerRunConfig(
                            verbose=False,
                            wait_for="css:body",
                            delay_before_return_html=1.5,
                            page_timeout=30000,
                        )
                        
                        async with AsyncWebCrawler() as crawler:
                            result = await crawler.arun(url, config)
                            
                            if hasattr(result, '__aiter__'):
                                async for r in result:
                                    markdown = getattr(r, 'markdown', None)
                                    if markdown and hasattr(markdown, 'raw_markdown'):
                                        content = markdown.raw_markdown[:4000]  # Limit content
                                        return await self._extract_with_llm(url, content)
                            else:
                                markdown = getattr(result, 'markdown', None)
                                if markdown and hasattr(markdown, 'raw_markdown'):
                                    content = markdown.raw_markdown[:4000]
                                    return await self._extract_with_llm(url, content)
                    
                    except Exception as e:
                        if attempt == self.config.retry_attempts - 1:
                            print(f"  ⚠️  Failed {url}: {str(e)[:100]}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                return None
        
        # Process all jobs concurrently
        tasks = [process_single_job(url) for url in job_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_jobs = []
        for result in results:
            if isinstance(result, dict) and result:
                successful_jobs.append(result)
                metrics.jobs_extracted += 1
        
        return successful_jobs
    
    async def _extract_with_llm(self, url: str, content: str) -> Dict[str, Any]:
        """Extract job data using LLM"""
        prompt = f"""
        Extract job information from this Capital One job posting.
        
        URL: {url}
        Content: {content}
        
        Return only valid JSON:
        {{
          "job_title": "...",
          "location": "...",
          "department": "...",
          "job_type": "Full-time/Part-time/Contract",
          "summary": "...",
          "qualifications": ["..."],
          "responsibilities": ["..."],
          "salary_range": "...",
          "benefits": "...",
          "url": "{url}"
        }}
        """
        
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=1024)
        )
        
        text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(text)
        data["url"] = url
        return data

# =================== MAIN EXECUTION ===================
async def main():
    """Enhanced main function with comprehensive error handling"""
    config = ScrapingConfig(
        max_pages=50,
        max_concurrent=6,  # Conservative for anti-detection
        request_delay_range=(1.5, 3.0),
    )
    
    print("="*80)
    print("🏦 ENHANCED CAPITAL ONE JOB SCRAPER v2.0")
    print("🛡️  Production-grade with anti-detection & fallback strategies")
    print("="*80)
    
    # Phase 1: Job Discovery
    crawler = EnhancedCapitalOneCrawler(config)
    job_urls = await crawler.discover_all_jobs()
    
    if not job_urls:
        print("❌ No jobs discovered. Exiting.")
        return
    
    # Save discovered URLs
    with open("capitalone_discovered_urls.json", "w") as f:
        json.dump(sorted(list(job_urls)), f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎯 JOB EXTRACTION PHASE - {len(job_urls)} jobs to process")
    print(f"{'='*80}")
    
    # Phase 2: Content Extraction
    processor = EnhancedJobProcessor(config)
    all_extracted_jobs = []
    
    # Process in batches to avoid overwhelming the target
    batch_size = 20
    total_batches = (len(job_urls) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(job_urls), batch_size), 1):
        batch_urls = list(job_urls)[i:i+batch_size]
        print(f"\n📦 Batch {batch_num}/{total_batches}: Processing {len(batch_urls)} jobs")
        
        try:
            batch_results = await processor.process_job_batch(batch_urls, crawler.metrics)
            all_extracted_jobs.extend(batch_results)
            print(f"  ✅ Successfully extracted {len(batch_results)}/{len(batch_urls)} jobs")
        except Exception as e:
            print(f"  ❌ Batch {batch_num} failed: {str(e)[:150]}")
        
        # Anti-detection delay between batches
        if batch_num < total_batches:
            delay = random.uniform(3.0, 8.0)
            print(f"  ⏳ Waiting {delay:.1f}s before next batch...")
            await asyncio.sleep(delay)
    
    # Phase 3: Results & Analysis
    output_file = f"capitalone_jobs_enhanced_{len(all_extracted_jobs)}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_extracted_jobs, f, indent=2, ensure_ascii=False)
    
    # Enhanced summary
    summary = {
        "execution_timestamp": time.time(),
        "discovery_metrics": {
            "total_urls_discovered": len(job_urls),
            "pages_attempted": crawler.metrics.pages_attempted,
            "pages_successful": crawler.metrics.pages_successful,
            "success_rate": crawler.metrics.success_rate(),
        },
        "extraction_metrics": {
            "jobs_processed": len(job_urls),
            "jobs_extracted": len(all_extracted_jobs),
            "extraction_rate": len(all_extracted_jobs) / len(job_urls),
        },
        "performance_metrics": {
            "total_runtime": time.time() - crawler.metrics.start_time,
            "jobs_per_second": len(all_extracted_jobs) / (time.time() - crawler.metrics.start_time),
            "anti_bot_detections": crawler.metrics.anti_bot_detections,
        },
        "sample_extracted_jobs": [
            {
                "title": job.get("job_title", "Unknown"),
                "location": job.get("location", "Unknown"),
                "department": job.get("department", "Unknown"),
                "url": job.get("url", "")
            }
            for job in all_extracted_jobs[:10]
        ]
    }
    
    with open("capitalone_enhanced_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("✅ ENHANCED SCRAPING COMPLETE!")
    print(f"💼 Successfully extracted {len(all_extracted_jobs)} jobs from {len(job_urls)} discovered")
    print(f"📊 Extraction rate: {len(all_extracted_jobs)/len(job_urls):.1%}")
    print(f"📁 Results: {output_file}")
    print(f"📈 Summary: capitalone_enhanced_summary.json")
    crawler.metrics.print_detailed_stats()
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())