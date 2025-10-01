# Workflow Automation Script

This script automates the complete job scraping and matching workflow by chaining together all the individual components.

## Overview

The workflow consists of 4 main steps:

1. **Parse Sitemap** - Extract job URLs from sitemap.xml
2. **Scrape Jobs** - Download markdown content from job URLs
3. **Process with LLM** - Convert markdown to structured JSON using Gemini
4. **Match Jobs** - Compare jobs against resume using cosine similarity (optional)

## Prerequisites

### Dependencies

Make sure you have all required Python packages installed:

```bash
pip install crawl4ai google-generativeai cloudscraper scikit-learn pandas numpy tqdm aiohttp
```

Or install from a requirements file if available.

### Environment Variables

Set the following environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
export GEMINI_MODEL="gemini-flash-lite-latest"  # Optional, has default
```

For testing, you can use:
```bash
export GEMINI_API_KEY="AIzaSyC0HG6WUm5QdVIidnGcwl4lwK2MH2zugJo"
export GEMINI_MODEL="gemini-flash-lite-latest"
```

## Usage

### Basic Usage (Scraping Only)

Scrape and process jobs without matching:

```bash
cd src
python3 workflow_automation.py \
  --sitemap-url "https://careers.google.com/jobs/sitemap" \
  --company-name "google" \
  --skip-matching
```

### Full Workflow with Matching

Include job matching against your resume:

```bash
python3 workflow_automation.py \
  --sitemap-url "https://jobs.apple.com/sitemap/sitemap-jobs-en-us.xml" \
  --company-name "apple" \
  --resume-path "../runtime_data/processed_resumes/my_resume.json"
```

### Custom Output Directory

Specify where all outputs should be saved:

```bash
python3 workflow_automation.py \
  --sitemap-url "https://www.wellsfargojobs.com/sitemap.xml" \
  --company-name "wellsfargo" \
  --output-dir "./my_output_folder" \
  --skip-matching
```

## Command-Line Options

- `--sitemap-url` (required): URL of the sitemap.xml to process
- `--company-name` (required): Name of the company (used for file naming)
- `--output-dir` (optional): Directory for all outputs (default: `./workflow_output`)
- `--resume-path` (optional): Path to your processed resume JSON for job matching
- `--skip-matching` (optional): Skip the final job matching step

## Output Structure

All outputs are organized in the specified output directory:

```
workflow_output/
├── sitemaps/
│   └── company_sitemap.txt          # Extracted job URLs
├── scraped_jobs/
│   └── company_jobs_timestamp.json  # Raw scraped markdown
├── processed_jobs/
│   └── company_processed.json       # Structured JSON from LLM
├── job_matches/
│   └── company_matches.csv          # Job match results (if enabled)
└── company_workflow_log.json        # Detailed workflow execution log
```

## Examples

### Example 1: Google Jobs

```bash
python3 workflow_automation.py \
  --sitemap-url "https://careers.google.com/jobs/sitemap" \
  --company-name "google"
```

### Example 2: Netflix with Matching

```bash
python3 workflow_automation.py \
  --sitemap-url "https://explore.jobs.netflix.net/careers/sitemap.xml" \
  --company-name "netflix" \
  --resume-path "../runtime_data/processed_resumes/jtarriela_resume.json"
```

### Example 3: Apple Jobs to Custom Directory

```bash
python3 workflow_automation.py \
  --sitemap-url "https://jobs.apple.com/sitemap/sitemap-jobs-en-us.xml" \
  --company-name "apple" \
  --output-dir "./apple_jobs_$(date +%Y%m%d)" \
  --skip-matching
```

## Workflow Details

### Step 1: Parse Sitemap

Uses `sitemap-extract/sitemap_extract.py` to:
- Download and parse the sitemap.xml
- Extract all job URLs
- Save to a text file (one URL per line)

### Step 2: Scrape Jobs

Uses `raw_job_scraper.py` to:
- Read URLs from the text file
- Download markdown content for each job posting
- Apply bot detection evasion
- Filter and clean markdown
- Save to JSON format for LLM processing

Configuration:
- Concurrent requests: 6
- Request delay: 1.0s
- User agent rotation: enabled
- Bot retry delays: 10s, 30s, 60s, 120s

### Step 3: Process with LLM

Uses `jobdata_markdown_json_concurrent.py` to:
- Process markdown in parallel using async
- Send to Gemini LLM for structured extraction
- Parse into standardized JSON schema
- Save processed jobs

Configuration:
- Max concurrent requests: 5
- Batch size: 20
- Output: Structured JSON with job details

### Step 4: Match Jobs (Optional)

Uses `cosine_similiarity.py` to:
- Load processed jobs and resume
- Generate embeddings using Gemini
- Calculate cosine similarity scores
- Rank jobs by match percentage
- Save matches to CSV

Only runs if:
- `--resume-path` is provided
- `--skip-matching` is NOT set

## Troubleshooting

### Missing Dependencies

If you get `ModuleNotFoundError`, install missing packages:

```bash
pip install [package-name]
```

### API Key Issues

If LLM processing fails:
1. Verify `GEMINI_API_KEY` is set: `echo $GEMINI_API_KEY`
2. Get a new key at: https://aistudio.google.com/app/apikey
3. Export it: `export GEMINI_API_KEY="your-key"`

### Sitemap Parsing Fails

Some sitemaps require special handling:
- Check if the URL is correct
- Try adding `--stealth` mode in sitemap_extract.py
- Some sites block scrapers (e.g., Goldman Sachs, Amazon)

### Bot Detection

If many jobs fail to scrape:
- Increase delays in ScrapingConfig
- Reduce concurrent requests
- Check for rate limiting
- Some companies have strong bot protection

### Out of Memory

If processing many jobs:
- Reduce batch_size in step 3
- Reduce max_concurrent_requests
- Process in smaller batches

## Advanced Usage

### Process Multiple Companies

Create a shell script to process multiple companies:

```bash
#!/bin/bash
companies=(
  "google|https://careers.google.com/jobs/sitemap"
  "apple|https://jobs.apple.com/sitemap/sitemap-jobs-en-us.xml"
  "netflix|https://explore.jobs.netflix.net/careers/sitemap.xml"
)

for entry in "${companies[@]}"; do
  IFS="|" read -r name url <<< "$entry"
  python3 workflow_automation.py \
    --sitemap-url "$url" \
    --company-name "$name" \
    --skip-matching
done
```

### Custom Scraper Filtering

The script creates a generic scraper that accepts all URLs from the sitemap. If you need custom filtering, you can:

1. Add a company-specific scraper to `raw_job_scraper.py`
2. The workflow will automatically use it if the company name matches

### Resume Processing

Before matching, process your resume:

```bash
python3 resume_parser.py --input my_resume.pdf --output ../runtime_data/processed_resumes/my_resume.json
```

## Integration with Existing Scripts

This workflow automation script is designed to work seamlessly with existing scripts:

- **sitemap_parse.py**: Now automated, but can still be run standalone
- **raw_job_scraper.py**: Used internally, existing company scrapers work
- **jobdata_markdown_json_concurrent.py**: Async processing for speed
- **cosine_similiarity.py**: Optional matching step

## Performance

Typical performance on a standard connection:

- **Step 1 (Sitemap)**: 10-60 seconds (depends on sitemap size)
- **Step 2 (Scraping)**: 5-10 minutes per 100 jobs
- **Step 3 (LLM)**: 2-5 minutes per 100 jobs (async processing)
- **Step 4 (Matching)**: 30-90 seconds per 100 jobs

Total for 100 jobs: ~10-20 minutes

## Cost Estimation

Using Gemini API:

- **Processing**: ~0.1 million tokens per 100 jobs
- **Matching**: ~0.05 million tokens per 100 jobs
- **Estimated cost**: Very low with Gemini free tier

Check current pricing at: https://ai.google.dev/pricing

## Workflow Log

Each run creates a detailed log in JSON format:

```json
[
  {
    "timestamp": "2025-01-15 10:30:00",
    "step": 1,
    "name": "Parse Sitemap",
    "status": "SUCCESS",
    "details": "Extracted 150 URLs to sitemap.txt"
  },
  ...
]
```

This log can be used for debugging and monitoring.

## Contributing

To extend this workflow:

1. Add new steps in the `WorkflowAutomation` class
2. Follow the pattern: `stepN_name(self) -> bool`
3. Use `_log_step()` for logging
4. Update the `run()` method to include new steps

## Support

For issues or questions:
1. Check the workflow log JSON file
2. Review individual script documentation
3. Test each step independently
4. Check environment variables and dependencies
