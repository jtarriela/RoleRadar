#!/usr/bin/env python3
"""
Usage example for the Career Site Scraper

This script demonstrates how to:
1. Create URL files from your sitemap data
2. Run scrapers for different companies
3. Process the results for LLM analysis
"""

import asyncio
import json
from pathlib import Path
from career_scraper import ScraperFactory, ScrapingConfig

def create_url_files_from_sitemap_data():
    """
    Create URL files from your sitemap data
    
    You'll need to extract URLs from your sitemap data and save them to text files
    """
    
    # Example Netflix URLs (you'd populate these from your actual data)
    netflix_urls = [
        "https://explore.jobs.netflix.net/careers/job/790298012086-expression-of-interest-feature-animation-art-department-burbank-california-united-states-of-america",
        "https://explore.jobs.netflix.net/careers/job/790298012335-consumer-insights-researcher-japan-tokyo-japan",
        "https://explore.jobs.netflix.net/careers/job/790298020581-distributed-systems-engineer-l5-data-platform-usa-remote",
        # ... add more URLs from your sitemap data
    ]
    
    # Save to file
    with open('netflix_urls.txt', 'w') as f:
        for url in netflix_urls:
            f.write(f"{url}\n")
    
    print("📁 Created netflix_urls.txt")
    
    # Similarly for other companies...
    # You can parse your sitemap data files and extract relevant URLs

async def scrape_single_company():
    """Example: Scrape a single company's jobs"""
    
    print("🚀 Single Company Scraping Example")
    print("=" * 50)
    
    # Create configuration
    config = ScrapingConfig(
        max_concurrent=5,
        delay_between_requests=0.8,
        max_retries=3,
        output_dir="job_data"
    )
    
    try:
        # Create Netflix scraper
        scraper = ScraperFactory.create_scraper(
            company='netflix',
            url_file_path='netflix_urls.txt',
            config=config
        )
        
        # Run the scraper
        results = await scraper.scrape_all_jobs()
        
        # Print results
        print(f"\n✅ Scraping Summary:")
        print(f"   Company: {results['company']}")
        print(f"   Successful: {results['successful_pages']}")
        print(f"   Failed: {results['failed_pages']}")
        print(f"   Time: {results['total_time_seconds']:.2f}s")
        print(f"   Output: {results['output_file']}")
        
        return results
        
    except FileNotFoundError:
        print("❌ netflix_urls.txt not found. Create it first!")
        return None

async def scrape_multiple_companies():
    """Example: Scrape multiple companies in sequence"""
    
    print("\n🚀 Multiple Companies Scraping Example")
    print("=" * 50)
    
    # Companies to scrape (ensure you have the URL files)
    companies = ['netflix', 'google', 'bankofamerica', 'capitalone']
    
    # Shared configuration
    config = ScrapingConfig(
        max_concurrent=8,
        delay_between_requests=1.0,
        output_dir="all_jobs_data"
    )
    
    results_summary = {}
    
    for company in companies:
        url_file = f"{company}_urls.txt"
        
        if not Path(url_file).exists():
            print(f"⚠️  Skipping {company}: {url_file} not found")
            continue
        
        try:
            print(f"\n📊 Processing {company.title()}...")
            
            # Create and run scraper
            scraper = ScraperFactory.create_scraper(company, url_file, config)
            results = await scraper.scrape_all_jobs()
            
            # Store results
            results_summary[company] = {
                'successful_pages': results['successful_pages'],
                'total_time': results['total_time_seconds'],
                'output_file': results['output_file']
            }
            
            print(f"   ✅ {company.title()}: {results['successful_pages']} jobs scraped")
            
        except Exception as e:
            print(f"   ❌ {company.title()} failed: {e}")
            results_summary[company] = {'error': str(e)}
    
    # Save overall summary
    with open('scraping_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📊 Overall Summary:")
    total_jobs = sum(r.get('successful_pages', 0) for r in results_summary.values())
    print(f"   Total jobs scraped: {total_jobs}")
    print(f"   Companies processed: {len(results_summary)}")
    print(f"   Summary saved to: scraping_summary.json")
    
    return results_summary

def prepare_for_llm_processing():
    """
    Example: How to prepare scraped data for LLM processing
    
    This shows how to load and format the cleaned markdown data
    for feeding into your resume matching LLM pipeline
    """
    
    print("\n🤖 Preparing Data for LLM Processing")
    print("=" * 50)
    
    job_data_dir = Path("job_data")  # or wherever you saved the data
    
    if not job_data_dir.exists():
        print("❌ No job data directory found. Run scraping first!")
        return
    
    # Find all markdown files
    markdown_files = list(job_data_dir.glob("*_markdown_only.json"))
    
    if not markdown_files:
        print("❌ No markdown files found!")
        return
    
    all_job_data = []
    
    for file_path in markdown_files:
        print(f"📄 Loading {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
            
        # Add company info to each job
        company = file_path.name.split('_')[0]
        for job in job_data:
            job['company'] = company
            all_job_data.append(job)
    
    print(f"📊 Loaded {len(all_job_data)} total jobs for LLM processing")
    
    # Save combined data for LLM
    output_file = "all_jobs_for_llm.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_job_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Combined data saved to: {output_file}")
    
    # Show example structure
    if all_job_data:
        print(f"\n📋 Example job structure:")
        example = all_job_data[0]
        print(f"   URL: {example['url'][:60]}...")
        print(f"   Company: {example['company']}")
        print(f"   Markdown length: {len(example['markdown'])} chars")
        print(f"   Preview: {example['markdown'][:150]}...")
    
    return all_job_data

async def main():
    """Main execution function"""
    
    print("🏢 Career Site Scraper - Usage Examples")
    print("=" * 60)
    
    # Step 1: Show available companies
    print(f"Available scrapers: {ScraperFactory.get_available_companies()}")
    
    # Step 2: Create example URL files (optional)
    print(f"\n📝 To get started:")
    print(f"   1. Extract URLs from your sitemap data")
    print(f"   2. Save them to text files (e.g., netflix_urls.txt)")
    print(f"   3. Run the scraping examples below")
    
    # Uncomment these to run examples:
    
    # Example 1: Single company
    # await scrape_single_company()
    
    # Example 2: Multiple companies
    # await scrape_multiple_companies()
    
    # Example 3: Prepare for LLM
    # prepare_for_llm_processing()
    
    print(f"\n✅ Examples complete! Check the output files.")

if __name__ == "__main__":
    asyncio.run(main())