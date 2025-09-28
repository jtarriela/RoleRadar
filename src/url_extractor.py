#!/usr/bin/env python3
"""
URL Extractor Utility

This script helps extract job URLs from your sitemap data files
and creates the text files needed for the scraper system.
"""

import re
from pathlib import Path
from typing import List, Dict

class URLExtractor:
    """Extract and filter URLs from sitemap data files"""
    
    def __init__(self):
        self.company_patterns = {
            'netflix': {
                'domain': 'explore.jobs.netflix.net',
                'job_patterns': [r'/careers/job/\d+'],
                'exclude_patterns': ['expression-of-interest']
            },
            'bankofamerica': {
                'domain': 'careers.bankofamerica.com',
                'job_patterns': [r'/job-details/', r'/jobs/', r'/career-opportunities/'],
                'exclude_patterns': ['/benefits', '/career-development', '/company', '/discover-your-career', '/errors', '/en-us$']
            },
            'google': {
                'domain': 'careers.google.com',
                'job_patterns': [r'/jobs/results/\d+'],
                'exclude_patterns': []
            },
            'capitalone': {
                'domain': 'capitalonecareers.com',
                'job_patterns': [r'/job/'],
                'exclude_patterns': [
                    '/10-things-', '/3-outstanding-', '/3-reasons-', '/3-ways-', 
                    '/4-steps-', '/4-tips-', '/5-job-', '/5-leadership-', 
                    '/5-lessons-', '/5-reasons-', '/6-job-', '/6-lessons-', 
                    '/6-resume-', '/7-resume-', '/7-tips-', '/a-day-in-',
                    '-blog-', '-article-'
                ]
            }
        }
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract all URLs from text content"""
        # Pattern to match URLs
        url_pattern = r'https?://[^\s<>"\']+(?:[^\s<>"\'.?!])'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        
        # Clean up URLs (remove trailing punctuation)
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation that might be captured
            url = re.sub(r'[,.;:!?]+$', '', url)
            cleaned_urls.append(url)
        
        return list(set(cleaned_urls))  # Remove duplicates
    
    def filter_urls_for_company(self, urls: List[str], company: str) -> List[str]:
        """Filter URLs for a specific company's job pages"""
        
        if company not in self.company_patterns:
            print(f"⚠️  Unknown company: {company}")
            return []
        
        patterns = self.company_patterns[company]
        domain = patterns['domain']
        job_patterns = patterns['job_patterns']
        exclude_patterns = patterns['exclude_patterns']
        
        filtered_urls = []
        
        for url in urls:
            # Must be from the correct domain
            if domain not in url:
                continue
            
            # Must match at least one job pattern
            if not any(re.search(pattern, url) for pattern in job_patterns):
                continue
            
            # Must not match any exclude pattern
            if any(pattern in url for pattern in exclude_patterns):
                continue
            
            filtered_urls.append(url)
        
        return filtered_urls
    
    def process_sitemap_file(self, file_path: str, company: str) -> List[str]:
        """Process a sitemap data file for a specific company"""
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return []
        
        print(f"📄 Processing {file_path.name} for {company}...")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract all URLs
        all_urls = self.extract_urls_from_text(content)
        print(f"   Found {len(all_urls)} total URLs")
        
        # Filter for job URLs
        job_urls = self.filter_urls_for_company(all_urls, company)
        print(f"   Filtered to {len(job_urls)} job URLs")
        
        return job_urls
    
    def create_url_files(self, sitemap_files: Dict[str, str], output_dir: str = "."):
        """
        Create URL files for multiple companies from sitemap data
        
        Args:
            sitemap_files: Dict mapping company name to sitemap file path
            output_dir: Directory to save URL files
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        for company, sitemap_file in sitemap_files.items():
            # Extract job URLs
            job_urls = self.process_sitemap_file(sitemap_file, company)
            
            if job_urls:
                # Save to file
                output_file = output_dir / f"{company}_urls.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for url in sorted(job_urls):
                        f.write(f"{url}\n")
                
                print(f"   ✅ Saved {len(job_urls)} URLs to {output_file}")
                results[company] = {
                    'url_count': len(job_urls),
                    'output_file': str(output_file)
                }
            else:
                print(f"   ❌ No job URLs found for {company}")
                results[company] = {'url_count': 0, 'error': 'No URLs found'}
        
        return results


def main():
    """Example usage of URL extractor"""
    
    print("🔍 URL Extractor for Career Site Scraper")
    print("=" * 50)
    
    extractor = URLExtractor()
    
    # Example: Process your sitemap data files
    # Update these paths to match your actual files
    sitemap_files = {
        'netflix': 'netflix_sitemap.txt',      # Your Netflix sitemap data
        'bankofamerica': 'bofa_sitemap.txt',   # Your Bank of America sitemap data  
        'google': 'google_sitemap.txt',        # Your Google sitemap data
        'capitalone': 'capitalone_sitemap.txt' # Your Capital One sitemap data
    }
    
    # Check which files exist
    existing_files = {}
    for company, file_path in sitemap_files.items():
        if Path(file_path).exists():
            existing_files[company] = file_path
        else:
            print(f"⚠️  {file_path} not found, skipping {company}")
    
    if not existing_files:
        print("\n📝 To use this extractor:")
        print("   1. Save your sitemap data to text files")
        print("   2. Update the sitemap_files dictionary above")
        print("   3. Run this script")
        print("\n   Expected file format: plain text with URLs")
        return
    
    # Process existing files
    print(f"\n🚀 Processing {len(existing_files)} sitemap files...")
    results = extractor.create_url_files(existing_files)
    
    # Summary
    print(f"\n📊 Summary:")
    total_urls = 0
    for company, result in results.items():
        count = result.get('url_count', 0)
        total_urls += count
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {company}: {count} job URLs")
    
    print(f"\n🎯 Total job URLs extracted: {total_urls}")
    print(f"✅ Ready for scraping! Use the generated *_urls.txt files")


def create_sample_data():
    """Create sample data files for testing"""
    
    print("📝 Creating sample sitemap data for testing...")
    
    # Sample Netflix data (from your provided data)
    netflix_sample = """
Source URL: https://explore.jobs.netflix.net/careers/sitemap.xml
Generated: 2025-09-27T17:37:21.727639
Total URLs: 539
--------------------------------------------------
https://explore.jobs.netflix.net/careers/job/790298012086-expression-of-interest-feature-animation-art-department-burbank-california-united-states-of-america?domain=netflix.com&microsite=netflix.com
https://explore.jobs.netflix.net/careers/job/790298012335-consumer-insights-researcher-japan-tokyo-japan?domain=netflix.com&microsite=netflix.com
https://explore.jobs.netflix.net/careers/job/790298020581-distributed-systems-engineer-l5-data-platform-usa-remote?domain=netflix.com&microsite=netflix.com
https://explore.jobs.netflix.net/careers/job/790298020849-distributed-systems-engineer-l4-cloud-engineering-usa-remote?domain=netflix.com&microsite=netflix.com
https://explore.jobs.netflix.net/careers/job/790298020874-software-engineer-l4-member-commerce-games-engineering-usa-remote?domain=netflix.com&microsite=netflix.com
"""
    
    # Sample Google data
    google_sample = """
Source URL: https://careers.google.com/jobs/sitemap
Generated: 2025-09-27T17:47:30.282096
Total URLs: 2859
--------------------------------------------------
https://careers.google.com/jobs/results/100033842055652038-product-strategy-and-operations-lead/
https://careers.google.com/jobs/results/100110454608536262-principal-architect/
https://careers.google.com/jobs/results/100147556415087302-technical-solutions-engineer/
https://careers.google.com/jobs/results/100149674102399686-technical-program-manager-ii/
https://careers.google.com/
"""
    
    # Save sample files
    with open('netflix_sitemap.txt', 'w') as f:
        f.write(netflix_sample)
    
    with open('google_sitemap.txt', 'w') as f:
        f.write(google_sample)
    
    print("✅ Created netflix_sitemap.txt and google_sitemap.txt")
    print("🚀 Now run the main() function to extract URLs!")


if __name__ == "__main__":
    # Uncomment to create sample data for testing
    # create_sample_data()
    
    main()