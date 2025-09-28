import os
import json
import asyncio
import time
import re
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# =================== CONFIGURATION ===================
@dataclass
class ScrapingConfig:
    """Configuration for scraping behavior"""
    max_concurrent: int = 8
    delay_between_requests: float = 0.5
    max_retries: int = 3
    timeout_seconds: int = 30
    output_dir: str = "scraped_jobs"
    max_chars: int = 4000  # Limit for LLM token efficiency
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True)

# =================== MARKDOWN CLEANER ===================
class MarkdownCleaner:
    """Strip URLs and navigation to save LLM tokens"""
    
    URL_PATTERNS = [
        r'\[[^\]]+\]\([^)]+\)',           # [text](url)
        r'<https?://[^>\s]+>',            # <https://...>
        r'https?://\S+',                  # raw URLs
        r'!\[[^\]]*\]\([^)]+\)',          # ![alt](image)
        r'\[.*?\]\(mailto:.*?\)',         # mailto links
    ]
    
    COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in URL_PATTERNS]
    
    @classmethod
    def clean_markdown(cls, markdown: str, max_chars: int = 4000) -> str:
        """Clean markdown for LLM processing"""
        if not markdown:
            return ""
        
        lines = []
        for line in markdown.splitlines():
            # Skip URL-containing lines
            if any(pattern.search(line) for pattern in cls.COMPILED_PATTERNS):
                continue
            
            line = line.strip()
            if line and not cls._is_navigation(line):
                lines.append(line)
        
        cleaned = "\n".join(lines)
        return cleaned[:max_chars] + "..." if len(cleaned) > max_chars else cleaned
    
    @classmethod
    def _is_navigation(cls, line: str) -> bool:
        """Check if line is navigation/menu content"""
        nav_words = ["home", "about", "contact", "login", "search", "menu", "©", "privacy"]
        return any(word in line.lower() for word in nav_words) and len(line) < 50

# =================== BASE SCRAPER ===================
class BaseJobScraper(ABC):
    """Base scraper that reads URL files and downloads markdown"""
    
    def __init__(self, url_file_path: str, config: ScrapingConfig = None):
        self.url_file_path = Path(url_file_path)
        self.config = config or ScrapingConfig()
        self.company_name = self.__class__.__name__.replace("Scraper", "").lower()
        
        if not self.url_file_path.exists():
            raise FileNotFoundError(f"URL file not found: {url_file_path}")
        
        # Load and filter URLs
        self.raw_urls = self._load_urls()
        self.job_urls = self._filter_job_urls(self.raw_urls)
        
        print(f"🏢 {self.company_name.title()} Scraper")
        print(f"   📁 Loaded: {len(self.raw_urls)} URLs")
        print(f"   💼 Filtered: {len(self.job_urls)} job URLs")
    
    def _load_urls(self) -> List[str]:
        """Load URLs from text file"""
        urls = []
        with open(self.url_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('http'):
                    urls.append(line)
        return urls
    
    @abstractmethod
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        """Filter URLs to job pages only - implement in child classes"""
        pass
    
    async def scrape_all(self) -> Dict[str, Any]:
        """Main scraping method"""
        start_time = time.time()
        print(f"\n🚀 Scraping {len(self.job_urls)} {self.company_name} jobs...")
        
        # Scrape with concurrency control
        scraped_data = await self._scrape_pages()
        
        # Save results
        results = self._save_results(scraped_data, start_time)
        
        print(f"✅ Complete! {results['successful']}/{len(self.job_urls)} pages")
        print(f"   ⏱️ Time: {results['time']:.1f}s")
        print(f"   📄 Output: {results['output_file']}")
        
        return results
    
    async def _scrape_pages(self) -> List[Dict[str, Any]]:
        """Scrape all pages with concurrency control"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=1.0,
            timeout=self.config.timeout_seconds
        )
        
        scraped_pages = []
        
        async with AsyncWebCrawler() as crawler:
            # Create all tasks
            tasks = [
                self._scrape_single_page(crawler, url, config, semaphore)
                for url in self.job_urls
            ]
            
            # Process in batches for progress updates
            batch_size = self.config.max_concurrent * 2
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                print(f"   📦 Batch {i//batch_size + 1}: {i+1}-{min(i+batch_size, len(tasks))}")
                
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result.get('success'):
                        scraped_pages.append(result)
                
                if i + batch_size < len(tasks):
                    await asyncio.sleep(self.config.delay_between_requests)
        
        return scraped_pages
    
    async def _scrape_single_page(self, crawler, url: str, config, semaphore) -> Dict[str, Any]:
        """Scrape single page with retries"""
        async with semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    result = await crawler.arun(url, config)
                    
                    # Extract markdown
                    markdown = ""
                    if hasattr(result, '__aiter__'):
                        async for r in result:
                            if hasattr(r, 'markdown') and r.markdown:
                                markdown = r.markdown.raw_markdown or ""
                                break
                    elif hasattr(result, 'markdown') and result.markdown:
                        markdown = result.markdown.raw_markdown or ""
                    
                    # Clean markdown
                    cleaned = MarkdownCleaner.clean_markdown(markdown, self.config.max_chars)
                    
                    return {
                        'success': True,
                        'url': url,
                        'markdown': cleaned,
                        'original_length': len(markdown),
                        'cleaned_length': len(cleaned)
                    }
                    
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        return {'success': False, 'url': url, 'error': str(e)}
                    await asyncio.sleep(0.5 * (attempt + 1))
    
    def _save_results(self, scraped_data: List[Dict[str, Any]], start_time: float) -> Dict[str, Any]:
        """Save results to JSON files"""
        timestamp = int(time.time())
        filename = f"{self.company_name}_jobs_{timestamp}"
        
        # Full results
        output_file = Path(self.config.output_dir) / f"{filename}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)
        
        # Just markdown for LLM
        markdown_file = Path(self.config.output_dir) / f"{filename}_for_llm.json"
        llm_data = [
            {'url': page['url'], 'markdown': page['markdown'], 'company': self.company_name}
            for page in scraped_data if page.get('success')
        ]
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            json.dump(llm_data, f, indent=2, ensure_ascii=False)
        
        # Summary
        successful = len([p for p in scraped_data if p.get('success')])
        total_time = time.time() - start_time
        
        return {
            'company': self.company_name,
            'successful': successful,
            'failed': len(self.job_urls) - successful,
            'time': total_time,
            'output_file': str(output_file),
            'llm_file': str(markdown_file)
        }

# =================== COMPANY SCRAPERS ===================

class NetflixScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        for url in urls:
            if '/careers/job/' in url and re.search(r'job/\d+', url):
                if 'expression-of-interest' not in url.lower():
                    job_urls.append(url)
        return job_urls

class GoogleScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        for url in urls:
            if '/jobs/results/' in url and re.search(r'/results/\d+', url):
                job_urls.append(url)
        return job_urls

class BankOfAmericaScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        exclude = ['/benefits', '/career-development', '/company', '/discover-your-career', '/errors']
        
        for url in urls:
            if any(pattern in url for pattern in ['/job-details/', '/jobs/', '/career-opportunities/']):
                if not any(ex in url for ex in exclude) and not url.endswith('/en-us'):
                    job_urls.append(url)
        return job_urls

class CapitalOneScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        exclude = [
            '/10-things-', '/3-outstanding-', '/3-reasons-', '/3-ways-', 
            '/4-steps-', '/4-tips-', '/5-job-', '/5-leadership-', 
            '/5-lessons-', '/5-reasons-', '/6-job-', '/6-lessons-', 
            '/6-resume-', '/7-resume-', '/7-tips-', '/a-day-in-',
            '-blog-', '-article-'
        ]
        
        for url in urls:
            if '/job/' in url:
                if not any(ex in url for ex in exclude):
                    job_urls.append(url)
        return job_urls

class AppleScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        for url in urls:
            if '/en-us/details/' in url and 'locationPicker' not in url:
                job_urls.append(url)
        return job_urls

class MicrosoftScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        for url in urls:
            if '/en-us/job/' in url or '/job/' in url:
                if 'search' not in url and 'filter' not in url:
                    job_urls.append(url)
        return job_urls

class AmazonScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        for url in urls:
            if '/jobs/' in url and '/job/' in url:
                if 'search' not in url and 'location' not in url:
                    job_urls.append(url)
        return job_urls

# =================== SCRAPER FACTORY ===================
class ScraperFactory:
    """Factory to create company scrapers"""
    
    SCRAPERS = {
        'netflix': NetflixScraper,
        'google': GoogleScraper,
        'bankofamerica': BankOfAmericaScraper,
        'bofa': BankOfAmericaScraper,
        'capitalone': CapitalOneScraper,
        'apple': AppleScraper,
        'microsoft': MicrosoftScraper,
        'amazon': AmazonScraper
    }
    
    @classmethod
    def create(cls, company: str, url_file: str, config: ScrapingConfig = None):
        """Create scraper for company"""
        company_key = company.lower()
        if company_key not in cls.SCRAPERS:
            available = ', '.join(cls.SCRAPERS.keys())
            raise ValueError(f"Unknown company '{company}'. Available: {available}")
        
        return cls.SCRAPERS[company_key](url_file, config)
    
    @classmethod
    def available_companies(cls) -> List[str]:
        return list(cls.SCRAPERS.keys())

# =================== MAIN EXECUTION ===================
async def scrape_company(company: str, url_file: str, config: ScrapingConfig = None):
    """Scrape a single company"""
    try:
        scraper = ScraperFactory.create(company, url_file, config)
        return await scraper.scrape_all()
    except Exception as e:
        print(f"❌ Error scraping {company}: {e}")
        return None

async def scrape_multiple_companies(company_files: Dict[str, str], config: ScrapingConfig = None):
    """Scrape multiple companies"""
    print("🚀 Multi-Company Job Scraper")
    print("=" * 40)
    
    results = {}
    total_jobs = 0
    
    for company, url_file in company_files.items():
        if not Path(url_file).exists():
            print(f"⚠️ Skipping {company}: {url_file} not found")
            continue
        
        print(f"\n📊 Processing {company.title()}...")
        result = await scrape_company(company, url_file, config)
        
        if result:
            results[company] = result
            total_jobs += result['successful']
            print(f"   ✅ {result['successful']} jobs scraped")
        else:
            print(f"   ❌ Failed")
    
    print(f"\n🎯 TOTAL: {total_jobs} jobs from {len(results)} companies")
    
    # Save combined results for LLM
    all_jobs = []
    for company, result in results.items():
        if result and result.get('llm_file'):
            with open(result['llm_file'], 'r') as f:
                jobs = json.load(f)
                all_jobs.extend(jobs)
    
    if all_jobs:
        output_file = "all_companies_jobs_for_llm.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_jobs, f, indent=2, ensure_ascii=False)
        print(f"💾 Combined LLM data: {output_file} ({len(all_jobs)} jobs)")
    
    return results

# =================== USAGE EXAMPLES ===================
async def main():
    """Example usage"""
    
    print("Available companies:", ScraperFactory.available_companies())
    
    # Configuration
    config = ScrapingConfig(
        max_concurrent=6,
        delay_between_requests=0.8,
        max_chars=4000,
        output_dir="job_data"
    )
    
    # Example 1: Single company
    result = await scrape_company('netflix', '/home/jd/proj/RoleRadar/test/explore_jobs_netflix_net_careers_sitemapxml.txt', config)
    
    # Example 2: Multiple companies
    company_files = {
        'netflix': '/home/jd/proj/RoleRadar/test/explore_jobs_netflix_net_careers_sitemapxml.txt',
        # 'google': 'google_urls.txt',
        # 'apple': 'apple_urls.txt',
        # 'microsoft': 'microsoft_urls.txt'
    }
    
    # Uncomment to run:
    # await scrape_multiple_companies(company_files, config)
    
    print("\n📝 To use:")
    print("1. Put your URL files in the current directory")
    print("2. Update company_files dict with your actual file names")
    print("3. Uncomment the scraping calls above")

if __name__ == "__main__":
    asyncio.run(main())