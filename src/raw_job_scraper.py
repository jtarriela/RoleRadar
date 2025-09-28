#raw_job_scraper.py
import os
import json
import asyncio
import time
import re
import random
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# This takes the text files of url and scrapes the job markdown.
# =================== CONFIGURATION ===================
@dataclass
class ScrapingConfig:
    """Configuration for scraping behavior"""
    max_concurrent: int = 8
    delay_between_requests: float = 0.5
    max_retries: int = 3
    timeout_seconds: int = 30
    output_dir: str = "scraped_jobs"
    max_chars: Optional[int] = None  # No limit - keep full content
    
    # Bot detection and evasion settings
    bot_retry_delays: List[float] = None  # [5, 15, 30, 60] - escalating delays
    max_bot_retries: int = 4
    user_agent_rotation: bool = True
    randomize_delays: bool = True
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True)
        if self.bot_retry_delays is None:
            self.bot_retry_delays = [5.0, 15.0, 30.0, 60.0]

# =================== USER AGENT ROTATION ===================
class UserAgentRotator:
    """Rotates user agents to appear more human"""
    
    USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        
        # Chrome on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        
        # Firefox on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        
        # Chrome on Linux
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        
        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
    ]
    
    def __init__(self):
        self.current_index = 0
        random.shuffle(self.USER_AGENTS)  # Randomize order
    
    def get_next_user_agent(self) -> str:
        """Get next user agent in rotation"""
        user_agent = self.USER_AGENTS[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.USER_AGENTS)
        return user_agent
    
    def get_random_user_agent(self) -> str:
        """Get random user agent"""
        return random.choice(self.USER_AGENTS)

# =================== BOT DETECTION ===================
class BotDetector:
    """Detects bot protection and CAPTCHA responses"""
    
    BOT_INDICATORS = [
        # General bot detection
        "temporary error",
        "solve a puzzle", 
        "security check",
        "confirm you are human",
        "complete the security check",
        "verify that you are not a bot",
        "you need to solve a puzzle",
        "are you a robot",
        "prove you're human",
        
        # Cloudflare
        "cloudflare",
        "checking your browser",
        "ray id",
        "performance & security by cloudflare",
        
        # Rate limiting
        "rate limited",
        "too many requests",
        "slow down",
        "request limit exceeded",
        "you have been rate limited",
        
        # CAPTCHA services
        "recaptcha",
        "hcaptcha", 
        "captcha",
        "i'm not a robot",
        
        # Access denied
        "access denied",
        "forbidden",
        "blocked",
        "not authorized",
        
        # Browser checks
        "enable javascript",
        "javascript required",
        "browser check",
        "loading...",
        "please wait",
        "redirecting",
        
        # Google Translate disable (Netflix specific)
        "google translate",
        "disable google translate",
    ]
    
    @classmethod
    def is_bot_detected(cls, content: str) -> bool:
        """Check if content indicates bot detection"""
        if not content:
            return False
            
        content_lower = content.lower()
        
        # Check for bot indicators
        for indicator in cls.BOT_INDICATORS:
            if indicator in content_lower:
                return True
        
        # Check for very short content (likely error pages)
        if len(content.strip()) < 200:
            return True
            
        # Check for suspicious language patterns
        language_indicators = ["العربية", "čeština", "dansk", "deutsch", "español", "français", "italiano"]
        if sum(1 for lang in language_indicators if lang in content_lower) >= 3:
            return True
            
        return False
    
    @classmethod
    def get_detected_type(cls, content: str) -> str:
        """Get the type of bot detection encountered"""
        content_lower = content.lower()
        
        if any(indicator in content_lower for indicator in ["cloudflare", "ray id"]):
            return "Cloudflare"
        elif any(indicator in content_lower for indicator in ["captcha", "recaptcha", "hcaptcha"]):
            return "CAPTCHA"
        elif any(indicator in content_lower for indicator in ["rate limit", "too many requests"]):
            return "Rate Limit"
        elif "google translate" in content_lower:
            return "Google Translate Block"
        else:
            return "General Bot Detection"

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
    def clean_markdown(cls, markdown: str, max_chars: Optional[int] = None) -> str:
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
        
        # Only truncate if max_chars is specified and content exceeds it
        if max_chars and len(cleaned) > max_chars:
            return cleaned[:max_chars] + "..."
        
        return cleaned
    
    @classmethod
    def _is_navigation(cls, line: str) -> bool:
        """Check if line is navigation/menu content"""
        nav_words = ["home", "about", "contact", "login", "search", "menu", "©", "privacy"]
        return any(word in line.lower() for word in nav_words) and len(line) < 50

# =================== BASE SCRAPER ===================
class BaseJobScraper(ABC):
    """Base scraper that reads URL files and downloads markdown"""
    
    def __init__(self, url_file_path: str, config: ScrapingConfig = None, max_urls: int = None):
        self.url_file_path = Path(url_file_path)
        self.config = config or ScrapingConfig()
        self.company_name = self.__class__.__name__.replace("Scraper", "").lower()
        self.max_urls = max_urls
        
        # Initialize user agent rotator
        self.user_agent_rotator = UserAgentRotator()
        
        # Bot detection stats
        self.bot_detections = 0
        self.successful_retries = 0
        
        if not self.url_file_path.exists():
            raise FileNotFoundError(f"URL file not found: {url_file_path}")
        
        # Load and filter URLs
        self.raw_urls = self._load_urls()
        self.job_urls = self._filter_job_urls(self.raw_urls)
        
        # Limit URLs if specified (for testing)
        if self.max_urls and len(self.job_urls) > self.max_urls:
            self.job_urls = self.job_urls[:self.max_urls]
            print(f"   🧪 Limited to {self.max_urls} URLs for testing")
        
        print(f"🏢 {self.company_name.title()} Scraper")
        print(f"   📁 Loaded: {len(self.raw_urls)} URLs")
        print(f"   💼 Filtered: {len(self.job_urls)} job URLs")
        print(f"   🤖 Bot evasion: {'Enabled' if self.config.user_agent_rotation else 'Disabled'}")
    
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
    
    def _get_random_delay(self, base_delay: float) -> float:
        """Add randomization to delays to appear more human"""
        if not self.config.randomize_delays:
            return base_delay
        
        # Add ±50% randomization
        variance = base_delay * 0.5
        return base_delay + random.uniform(-variance, variance)
    
    async def scrape_all(self) -> Dict[str, Any]:
        """Main scraping method"""
        start_time = time.time()
        print(f"\n🚀 Scraping {len(self.job_urls)} {self.company_name} jobs...")
        
        # Scrape with concurrency control
        scraped_data = await self._scrape_pages()
        
        # Save results
        results = self._save_results(scraped_data, start_time)
        
        # Print bot detection stats
        if self.bot_detections > 0:
            print(f"   🤖 Bot detections: {self.bot_detections}")
            print(f"   ✅ Successful retries: {self.successful_retries}")
        
        print(f"✅ Complete! {results['successful']}/{len(self.job_urls)} pages")
        print(f"   ⏱️ Time: {results['time']:.1f}s")
        print(f"   📄 Output: {results['output_file']}")
        
        return results
    
    async def _scrape_pages(self) -> List[Dict[str, Any]]:
        """Scrape all pages with concurrency control and failed URL retry system"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        scraped_pages = []
        failed_urls = []  # Track URLs that failed due to bot detection
        
        print(f"🔄 Starting initial pass through {len(self.job_urls)} URLs...")
        
        async with AsyncWebCrawler() as crawler:
            # FIRST PASS: Process all URLs with normal retry logic
            tasks = [
                self._scrape_single_page_with_evasion(crawler, url, semaphore)
                for url in self.job_urls
            ]
            
            # Process in batches for progress updates
            batch_size = self.config.max_concurrent * 2
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                print(f"   📦 Batch {i//batch_size + 1}: {i+1}-{min(i+batch_size, len(tasks))}")
                
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result, original_url in zip(results, self.job_urls[i:i+batch_size]):
                    if isinstance(result, dict):
                        if result.get('success'):
                            scraped_pages.append(result)
                        elif 'Bot detection' in result.get('error', '') or 'rate limit' in result.get('error', '').lower():
                            failed_urls.append(original_url)
                            print(f"   📋 Marked for retry: {original_url}")
                
                if i + batch_size < len(tasks):
                    delay = self._get_random_delay(self.config.delay_between_requests)
                    await asyncio.sleep(delay)
        
        # SECOND PASS: Retry failed URLs with more conservative settings
        if failed_urls:
            print(f"\n🔄 Second pass: Retrying {len(failed_urls)} failed URLs...")
            print("   ⚙️ Using more conservative settings...")
            
            # Wait longer before starting retry pass
            retry_wait = 60.0  # 1 minute cooldown
            print(f"   ⏳ Waiting {retry_wait}s for rate limits to reset...")
            await asyncio.sleep(retry_wait)
            
            # More conservative retry settings
            retry_semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent // 3))  # Much lower concurrency
            
            retry_tasks = [
                self._scrape_single_page_conservative(crawler, url, retry_semaphore)
                for url in failed_urls
            ]
            
            # Process retry batch with longer delays
            for i in range(0, len(retry_tasks), 5):  # Smaller batches
                batch = retry_tasks[i:i + 5]
                print(f"   🔄 Retry batch {i//5 + 1}: {i+1}-{min(i+5, len(retry_tasks))}")
                
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                successful_retries = 0
                for result in results:
                    if isinstance(result, dict) and result.get('success'):
                        scraped_pages.append(result)
                        successful_retries += 1
                        self.successful_retries += 1
                
                print(f"   ✅ {successful_retries}/{len(batch)} successful in retry batch")
                
                # Longer delay between retry batches
                if i + 5 < len(retry_tasks):
                    retry_delay = self._get_random_delay(self.config.delay_between_requests * 3)
                    await asyncio.sleep(retry_delay)
        
        return scraped_pages
    
    async def _scrape_single_page_conservative(self, crawler, url: str, semaphore) -> Dict[str, Any]:
        """More conservative scraping for retry attempts"""
        async with semaphore:
            
            # Much longer delays and more conservative settings for retries
            conservative_config = {
                'bot_retry_delays': [30.0, 90.0, 180.0],  # Longer delays
                'max_bot_retries': 2,  # Fewer attempts
                'base_delay': 3.0      # Longer base delay
            }
            
            # Regular retries for network issues
            for attempt in range(2):  # Fewer regular retries
                try:
                    result = await self._attempt_scrape_with_rotation(crawler, url)
                    
                    # Check for bot detection
                    if result.get('success') and BotDetector.is_bot_detected(result.get('markdown', '')):
                        detection_type = BotDetector.get_detected_type(result.get('markdown', ''))
                        print(f"   🤖 Bot detected (retry): {url}: {detection_type}")
                        
                        # Try conservative bot evasion
                        bot_result = await self._handle_bot_detection_conservative(
                            crawler, url, detection_type, conservative_config
                        )
                        if bot_result:
                            print(f"   ✅ Conservative evasion successful: {url}")
                            return bot_result
                        
                        # If still failing, return failure
                        return {
                            'success': False, 
                            'url': url, 
                            'error': f'Persistent bot detection: {detection_type}',
                            'retry_pass': True
                        }
                    
                    # Normal successful result
                    if result.get('success'):
                        return result
                    
                except Exception as e:
                    if attempt == 1:  # Last attempt
                        return {'success': False, 'url': url, 'error': f'Conservative retry failed: {str(e)}'}
                    
                    # Add delay between retries
                    delay = self._get_random_delay(conservative_config['base_delay'] * (attempt + 1))
                    await asyncio.sleep(delay)
            
            return {'success': False, 'url': url, 'error': 'Conservative retries exhausted'}
    
    async def _handle_bot_detection_conservative(self, crawler, url: str, detection_type: str, config: Dict) -> Optional[Dict[str, Any]]:
        """More conservative bot detection handling for retry pass"""
        
        print(f"   🔄 Conservative bot evasion: {url}")
        
        for retry_attempt in range(config['max_bot_retries']):
            delay = config['bot_retry_delays'][min(retry_attempt, len(config['bot_retry_delays']) - 1)]
            
            print(f"   ⏳ Conservative wait {delay}s (attempt {retry_attempt + 1}/{config['max_bot_retries']})")
            await asyncio.sleep(delay)
            
            try:
                # Use a random user agent for each retry
                user_agent = self.user_agent_rotator.get_random_user_agent()
                
                config_obj = CrawlerRunConfig(
                    verbose=False,
                    wait_for="css:body",
                    delay_before_return_html=4.0,  # Even longer delay
                    user_agent=user_agent
                )
                
                result = await crawler.arun(url, config_obj)
                
                # Extract markdown
                markdown = ""
                if hasattr(result, '__aiter__'):
                    async for r in result:
                        if hasattr(r, 'markdown') and r.markdown:
                            markdown = r.markdown.raw_markdown or ""
                            break
                elif hasattr(result, 'markdown') and result.markdown:
                    markdown = result.markdown.raw_markdown or ""
                
                # Check if we still have bot detection
                if not BotDetector.is_bot_detected(markdown):
                    cleaned = MarkdownCleaner.clean_markdown(markdown, self.config.max_chars)
                    
                    return {
                        'success': True,
                        'url': url,
                        'markdown': cleaned,
                        'original_length': len(markdown),
                        'cleaned_length': len(cleaned),
                        'user_agent': user_agent,
                        'conservative_retry': True,
                        'bot_evasion_attempts': retry_attempt + 1
                    }
                else:
                    print(f"   🤖 Still detected in conservative retry {retry_attempt + 1}")
                    
            except Exception as e:
                print(f"   ❌ Conservative retry {retry_attempt + 1} failed: {e}")
        
        print(f"   ❌ Conservative bot evasion failed: {url}")
        return None
    
    async def _scrape_single_page_with_evasion(self, crawler, url: str, semaphore) -> Dict[str, Any]:
        """Scrape single page with bot detection and evasion"""
        async with semaphore:
            
            # Regular retries for network issues
            for attempt in range(self.config.max_retries):
                try:
                    result = await self._attempt_scrape_with_rotation(crawler, url)
                    
                    # Check for bot detection
                    if result.get('success') and BotDetector.is_bot_detected(result.get('markdown', '')):
                        detection_type = BotDetector.get_detected_type(result.get('markdown', ''))
                        print(f"   🤖 Bot detected for {url}: {detection_type}")
                        self.bot_detections += 1
                        
                        # Try bot evasion retries
                        bot_result = await self._handle_bot_detection(crawler, url, detection_type)
                        if bot_result:
                            self.successful_retries += 1
                            return bot_result
                        
                        # If bot evasion failed, return the bot detection result
                        return {
                            'success': False, 
                            'url': url, 
                            'error': f'Bot detection: {detection_type}',
                            'markdown': result.get('markdown', '')
                        }
                    
                    # Normal successful result
                    if result.get('success'):
                        return result
                    
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        return {'success': False, 'url': url, 'error': str(e)}
                    
                    # Add delay between retries
                    delay = self._get_random_delay(0.5 * (attempt + 1))
                    await asyncio.sleep(delay)
            
            return {'success': False, 'url': url, 'error': 'Max retries exceeded'}
    
    async def _attempt_scrape_with_rotation(self, crawler, url: str) -> Dict[str, Any]:
        """Attempt to scrape with user agent rotation"""
        
        # Get user agent
        if self.config.user_agent_rotation:
            user_agent = self.user_agent_rotator.get_next_user_agent()
        else:
            user_agent = None
        
        # Create config with user agent
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=1.0,
            user_agent=user_agent
        )
        
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
        
        # Clean markdown (keep full content unless max_chars specified)
        cleaned = MarkdownCleaner.clean_markdown(markdown, self.config.max_chars)
        
        return {
            'success': True,
            'url': url,
            'markdown': cleaned,
            'original_length': len(markdown),
            'cleaned_length': len(cleaned),
            'user_agent': user_agent
        }
    
    async def _handle_bot_detection(self, crawler, url: str, detection_type: str) -> Optional[Dict[str, Any]]:
        """Handle bot detection with escalating delays and user agent changes"""
        
        print(f"   🔄 Attempting bot evasion for {url}")
        
        for retry_attempt in range(self.config.max_bot_retries):
            delay = self.config.bot_retry_delays[min(retry_attempt, len(self.config.bot_retry_delays) - 1)]
            
            print(f"   ⏳ Waiting {delay}s before retry {retry_attempt + 1}/{self.config.max_bot_retries}")
            await asyncio.sleep(delay)
            
            try:
                # Use a random user agent for each retry
                user_agent = self.user_agent_rotator.get_random_user_agent()
                
                config = CrawlerRunConfig(
                    verbose=False,
                    wait_for="css:body",
                    delay_before_return_html=2.0,  # Longer delay
                    user_agent=user_agent
                )
                
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
                
                # Check if we still have bot detection
                if not BotDetector.is_bot_detected(markdown):
                    cleaned = MarkdownCleaner.clean_markdown(markdown, self.config.max_chars)
                    print(f"   ✅ Bot evasion successful for {url}")
                    
                    return {
                        'success': True,
                        'url': url,
                        'markdown': cleaned,
                        'original_length': len(markdown),
                        'cleaned_length': len(cleaned),
                        'user_agent': user_agent,
                        'bot_evasion_attempts': retry_attempt + 1
                    }
                else:
                    print(f"   🤖 Still detected after retry {retry_attempt + 1}")
                    
            except Exception as e:
                print(f"   ❌ Retry {retry_attempt + 1} failed: {e}")
        
        print(f"   ❌ Bot evasion failed for {url}")
        return None
    
    def _save_results(self, scraped_data: List[Dict[str, Any]], start_time: float) -> Dict[str, Any]:
        """Save results to JSON files"""
        timestamp = int(time.time())
        filename = f"{self.company_name}_jobs_{timestamp}"
        
        # Full results
        output_file = Path(self.config.output_dir) / f"{filename}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)
        
        # Just markdown for LLM (filter out bot detection failures)
        markdown_file = Path(self.config.output_dir) / f"{filename}_for_llm.json"
        llm_data = []
        
        for page in scraped_data:
            if page.get('success') and page.get('markdown'):
                # Double-check no bot detection slipped through
                if not BotDetector.is_bot_detected(page['markdown']):
                    llm_data.append({
                        'url': page['url'], 
                        'markdown': page['markdown'], 
                        'company': self.company_name
                    })
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            json.dump(llm_data, f, indent=2, ensure_ascii=False)
        
        # Summary
        successful = len([p for p in scraped_data if p.get('success')])
        total_time = time.time() - start_time
        
        return {
            'company': self.company_name,
            'successful': successful,
            'failed': len(self.job_urls) - successful,
            'bot_detections': self.bot_detections,
            'successful_retries': self.successful_retries,
            'clean_jobs_for_llm': len(llm_data),
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
            if any(pattern in url for pattern in ['/job-detail/']):
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

class WellsFargoScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        for url in urls:
            # Wells Fargo job URLs pattern: https://www.wellsfargojobs.com/en/jobs/r-{number}/job-title/
            if '/jobs/r-' in url and re.search(r'/jobs/r-\d+', url):
                # Exclude non-job pages
                exclude_patterns = [
                    '/search', '/filter', '/location', '/category',
                    '/benefits', '/career-development', '/how-to-apply',
                    '/job-alerts', '/student-programs'
                ]
                if not any(pattern in url.lower() for pattern in exclude_patterns):
                    job_urls.append(url)
        return job_urls

class MetaScraper(BaseJobScraper):
    def _filter_job_urls(self, urls: List[str]) -> List[str]:
        job_urls = []
        for url in urls:
            # Meta job URLs pattern: https://www.metacareers.com/jobs/{numeric_id}
            if '/jobs/' in url and re.search(r'/jobs/\d+', url):
                # Exclude non-job pages
                exclude_patterns = [
                    '/search', '/filter', '/location', '/category',
                    '/internships-', '/university-', '/how-to-apply'
                ]
                if not any(pattern in url.lower() for pattern in exclude_patterns):
                    job_urls.append(url)
        return job_urls

# =================== SCRAPER FACTORY ===================
class ScraperFactory:
    """Factory to create company scrapers"""
    
    SCRAPERS = {
        'netflix': NetflixScraper,
        'google': GoogleScraper,
        'bofa': BankOfAmericaScraper,
        'capitalone': CapitalOneScraper,
        'apple': AppleScraper,
        'microsoft': MicrosoftScraper,
        'meta': MetaScraper,
        'wellsfargo': WellsFargoScraper
    }
    
    @classmethod
    def create(cls, company: str, url_file: str, config: ScrapingConfig = None, max_urls: int = None):
        """Create scraper for company"""
        company_key = company.lower()
        if company_key not in cls.SCRAPERS:
            available = ', '.join(cls.SCRAPERS.keys())
            raise ValueError(f"Unknown company '{company}'. Available: {available}")
        
        return cls.SCRAPERS[company_key](url_file, config, max_urls)
    
    @classmethod
    def available_companies(cls) -> List[str]:
        return list(cls.SCRAPERS.keys())

# =================== MAIN EXECUTION ===================
async def scrape_company(company: str, url_file: str, config: ScrapingConfig = None, max_urls: int = None):
    """Scrape a single company"""
    try:
        scraper = ScraperFactory.create(company, url_file, config, max_urls)
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
    # This takes the text files of url and scrapes the job markdown.
    
    print("Available companies:", ScraperFactory.available_companies())
    
    # Configuration with bot evasion
    config = ScrapingConfig(
        max_concurrent=6,  # Reduced to be less aggressive
        delay_between_requests=1.0,  # Increased base delay
        max_chars=100000,
        output_dir="job_data",
        
        # Bot evasion settings
        user_agent_rotation=True,
        randomize_delays=True,
        bot_retry_delays=[10.0, 30.0, 60.0, 120.0],  # Longer delays
        max_bot_retries=3
    )
    '''
        SCRAPERS = {
        'netflix': NetflixScraper,
        'google': GoogleScraper,
        'bofa': BankOfAmericaScraper,
        'capitalone': CapitalOneScraper,
        'apple': AppleScraper,
        'microsoft': MicrosoftScraper,
        'meta': MetaScraper,
        'wellsfargo': WellsFargoScraper
        '''
    # Example: Netflix with bot evasion
    result = await scrape_company('netflix', '../test/netflix.txt', config, max_urls=10000)  # Start small to test
    result = await scrape_company('google', '../test/google.txt', config, max_urls=10000)  # Start small to test
    result = await scrape_company('bofa', '../test/bofa.txt', config, max_urls=10000)  # Start small to test
    result = await scrape_company('capitalone', '../test/c1.txt', config, max_urls=10000)  # Start small to test
    # result = await scrape_company('apple', '../test/apple.txt', config, max_urls=10000)  # Start small to test
    # result = await scrape_company('microsoft', '../test/microsoft.txt', config, max_urls=10000)  # Start small to test
    result = await scrape_company('meta', '../test/meta.txt', config, max_urls=10000)  # Start small to test
    result = await scrape_company('wellsfargo', '../test/wellsfargo.txt', config, max_urls=10000)  # Start small to test

    
    print("\n📝 Bot Evasion Features:")
    print("✅ User agent rotation (10 different browsers)")
    print("✅ Bot detection with intelligent retries")
    print("✅ Escalating delays (10s → 30s → 60s → 120s)")
    print("✅ Random timing to appear human")
    print("✅ Automatic filtering of CAPTCHA pages")

if __name__ == "__main__":
    asyncio.run(main())