import re
import asyncio
from typing import Dict, List, Set
from urllib.parse import urljoin
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

class CapitalOneSitemapAnalyzer:
    """
    Analyzes Capital One's sitemap structure and categorizes URLs correctly
    """
    
    def __init__(self):
        self.url_categories = {
            'individual_jobs': set(),
            'category_pages': set(), 
            'location_pages': set(),
            'employment_filters': set(),
            'other_pages': set()
        }
    
    def classify_url(self, url: str) -> str:
        """
        Classify Capital One URLs into different types
        """
        # Individual job posting pattern: /job/location/title/company_id/job_id
        if re.match(r'.*?/job/[^/]+/[^/]+/\d+/\d+/?$', url):
            return 'individual_jobs'
        
        # Category pages: /category/job-type/company_id/category_id/page
        elif '/category/' in url and '-jobs/' in url:
            return 'category_pages'
        
        # Location pages: /location/place-jobs/company_id/location_ids/level
        elif '/location/' in url and '-jobs/' in url:
            return 'location_pages'
        
        # Employment filter combinations: /employment/location-category-jobs/...
        elif '/employment/' in url and '-jobs/' in url:
            return 'employment_filters'
        
        # Everything else
        else:
            return 'other_pages'
    
    async def analyze_sitemap(self, sitemap_url: str = "https://www.capitalonecareers.com/sitemap.xml") -> Dict:
        """
        Fetch and analyze the sitemap structure
        """
        print(f"🔍 Analyzing sitemap: {sitemap_url}")
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=1.0,
            page_timeout=20000,
        )
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(sitemap_url, config)
            
            # Get content from result
            if hasattr(result, '__aiter__'):
                async for r in result:
                    content = getattr(r, 'html', '') or getattr(r, 'markdown', '') or str(r)
                    break
            else:
                content = getattr(result, 'html', '') or getattr(result, 'markdown', '') or str(result)
            
            # Extract all URLs from sitemap
            url_matches = re.findall(r'<loc>(.*?)</loc>', content)
            
            # Classify each URL
            for url in url_matches:
                category = self.classify_url(url)
                self.url_categories[category].add(url)
            
            # Generate analysis report
            analysis = {
                'total_urls': len(url_matches),
                'breakdown': {
                    category: len(urls) for category, urls in self.url_categories.items()
                },
                'findings': self._generate_findings(),
                'sample_urls': self._get_sample_urls(),
                'recommended_strategy': self._get_recommendation()
            }
            
            return analysis
    
    def _generate_findings(self) -> List[str]:
        """Generate insights from the analysis"""
        findings = []
        
        individual_jobs = len(self.url_categories['individual_jobs'])
        category_pages = len(self.url_categories['category_pages'])
        location_pages = len(self.url_categories['location_pages'])
        
        if individual_jobs == 0:
            findings.append("❌ CRITICAL: No individual job URLs found in sitemap")
            findings.append("🔍 Sitemap contains only category/filter pages, not job postings")
        
        if category_pages > 0:
            findings.append(f"📂 Found {category_pages} job category pages (e.g., 'technology-jobs')")
        
        if location_pages > 0:
            findings.append(f"📍 Found {location_pages} location-based job pages")
        
        findings.append("💡 Strategy needed: Crawl category pages to find actual job URLs")
        
        return findings
    
    def _get_sample_urls(self) -> Dict[str, List[str]]:
        """Get sample URLs for each category"""
        samples = {}
        for category, urls in self.url_categories.items():
            samples[category] = list(urls)[:3]  # First 3 URLs of each type
        return samples
    
    def _get_recommendation(self) -> str:
        """Provide strategic recommendation"""
        if len(self.url_categories['individual_jobs']) == 0:
            return """
            RECOMMENDED STRATEGY:
            1. ❌ Don't rely on sitemap for individual jobs
            2. ✅ Use category pages as starting points
            3. ✅ Crawl category pages to extract individual job URLs
            4. ✅ Focus on high-volume categories first
            """
        else:
            return "✅ Sitemap contains individual job URLs - proceed with direct extraction"

async def demonstrate_analysis():
    """
    Demonstrate the sitemap analysis on Capital One
    """
    analyzer = CapitalOneSitemapAnalyzer()
    
    print("🔬 CAPITAL ONE SITEMAP ANALYSIS")
    print("="*50)
    
    analysis = await analyzer.analyze_sitemap()
    
    print(f"\n📊 SITEMAP BREAKDOWN:")
    print(f"Total URLs found: {analysis['total_urls']}")
    for category, count in analysis['breakdown'].items():
        print(f"  • {category.replace('_', ' ').title()}: {count}")
    
    print(f"\n🔍 KEY FINDINGS:")
    for finding in analysis['findings']:
        print(f"  {finding}")
    
    print(f"\n🎯 SAMPLE URLs BY CATEGORY:")
    for category, sample_urls in analysis['sample_urls'].items():
        if sample_urls:  # Only show categories with URLs
            print(f"\n{category.replace('_', ' ').title()}:")
            for url in sample_urls:
                print(f"  • {url}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(analysis['recommended_strategy'])
    
    return analysis

# Enhanced URL extraction strategy based on findings
class CapitalOneJobExtractor:
    """
    Extracts individual job URLs from Capital One category pages
    """
    
    def __init__(self):
        self.job_url_pattern = re.compile(r'/job/[^/]+/[^/]+/\d+/\d+')
    
    async def extract_jobs_from_category_page(self, category_url: str) -> Set[str]:
        """
        Extract individual job URLs from a category page
        """
        print(f"🔍 Extracting jobs from: {category_url}")
        
        config = CrawlerRunConfig(
            verbose=False,
            wait_for="css:body",
            delay_before_return_html=2.0,
            page_timeout=30000,
        )
        
        job_urls = set()
        
        async with AsyncWebCrawler() as crawler:
            try:
                result = await crawler.arun(category_url, config)
                
                # Get HTML content
                if hasattr(result, '__aiter__'):
                    async for r in result:
                        html = getattr(r, 'html', '')
                        break
                else:
                    html = getattr(result, 'html', '')
                
                # Extract job URLs using multiple patterns
                patterns = [
                    r'href=["\']([^"\']*?/job/[^"\']*?)["\']',
                    r'<a[^>]+href=["\']([^"\']*?/job/[^"\']*?)["\']',
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, html, re.IGNORECASE)
                    for match in matches:
                        url = match.group(1)
                        if url.startswith('/'):
                            url = urljoin(category_url, url)
                        
                        # Only keep individual job URLs, not category pages
                        if self.job_url_pattern.search(url):
                            job_urls.add(url)
                
                print(f"  ✅ Found {len(job_urls)} individual job URLs")
                
            except Exception as e:
                print(f"  ❌ Error processing {category_url}: {str(e)[:100]}")
        
        return job_urls
    
    async def extract_from_multiple_categories(self, category_urls: List[str], max_concurrent: int = 3) -> Set[str]:
        """
        Extract job URLs from multiple category pages concurrently
        """
        print(f"🚀 Processing {len(category_urls)} category pages...")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_category(url):
            async with semaphore:
                return await self.extract_jobs_from_category_page(url)
        
        # Process all categories concurrently
        tasks = [process_category(url) for url in category_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all job URLs
        all_jobs = set()
        for result in results:
            if isinstance(result, set):
                all_jobs.update(result)
        
        print(f"🎯 Total individual jobs found: {len(all_jobs)}")
        return all_jobs

# Main execution function
async def main():
    """
    Complete analysis and extraction workflow
    """
    print("🏦 CAPITAL ONE JOB DISCOVERY - INTELLIGENT APPROACH")
    print("="*60)
    
    # Phase 1: Analyze sitemap structure
    analysis = await demonstrate_analysis()
    
    # Phase 2: Extract jobs from category pages (if needed)
    if analysis['breakdown']['individual_jobs'] == 0:
        print(f"\n{'='*60}")
        print("🎯 PHASE 2: EXTRACTING FROM CATEGORY PAGES")
        print("="*60)
        
        extractor = CapitalOneJobExtractor()
        
        # Get category URLs from analysis
        category_urls = list(analysis['sample_urls']['category_pages'])
        
        if category_urls:
            job_urls = await extractor.extract_from_multiple_categories(category_urls[:5])  # Test with first 5
            
            print(f"\n✅ DISCOVERY COMPLETE!")
            print(f"📊 Found {len(job_urls)} individual job postings")
            
            # Save results
            with open("capital_one_individual_jobs.txt", "w") as f:
                for url in sorted(job_urls):
                    f.write(f"{url}\n")
            
            print(f"💾 Saved job URLs to: capital_one_individual_jobs.txt")
            
            # Show samples
            print(f"\n🔍 SAMPLE INDIVIDUAL JOB URLs:")
            for i, url in enumerate(sorted(job_urls)[:10], 1):
                print(f"  {i}. {url}")
        
        else:
            print("❌ No category pages found to extract from")

if __name__ == "__main__":
    asyncio.run(main())