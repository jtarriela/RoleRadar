# Workflow Automation Quick Start

## What is this?

An automated script that combines all job scraping steps into one command:

1. **Parse Sitemap** → Extract job URLs from sitemap.xml
2. **Scrape Jobs** → Download markdown from all job URLs  
3. **Process with LLM** → Convert markdown to structured JSON
4. **Match Jobs** → Compare with resume (optional)

## Quick Setup

```bash
# 1. Set environment variables
export GEMINI_API_KEY="AIzaSyC0HG6WUm5QdVIidnGcwl4lwK2MH2zugJo"
export GEMINI_MODEL="gemini-flash-lite-latest"

# 2. Make sure dependencies are installed (if needed)
pip install crawl4ai google-generativeai cloudscraper scikit-learn pandas numpy tqdm aiohttp

# 3. Run the workflow
cd src
python3 workflow_automation.py \
  --sitemap-url "https://careers.google.com/jobs/sitemap" \
  --company-name "google" \
  --skip-matching
```

## Basic Usage

```bash
# Just scrape and process (no matching)
python3 workflow_automation.py \
  --sitemap-url "URL_HERE" \
  --company-name "company_name" \
  --skip-matching

# Full workflow with job matching
python3 workflow_automation.py \
  --sitemap-url "URL_HERE" \
  --company-name "company_name" \
  --resume-path "path/to/resume.json"
```

## Example Companies

```bash
# Google
python3 workflow_automation.py \
  --sitemap-url "https://careers.google.com/jobs/sitemap" \
  --company-name "google" --skip-matching

# Apple
python3 workflow_automation.py \
  --sitemap-url "https://jobs.apple.com/sitemap/sitemap-jobs-en-us.xml" \
  --company-name "apple" --skip-matching

# Netflix  
python3 workflow_automation.py \
  --sitemap-url "https://explore.jobs.netflix.net/careers/sitemap.xml" \
  --company-name "netflix" --skip-matching

# Wells Fargo
python3 workflow_automation.py \
  --sitemap-url "https://www.wellsfargojobs.com/sitemap.xml" \
  --company-name "wellsfargo" --skip-matching
```

## Output

All files go to `workflow_output/` by default:

```
workflow_output/
├── sitemaps/company_sitemap.txt          # Job URLs
├── scraped_jobs/company_jobs_*.json      # Raw markdown
├── processed_jobs/company_processed.json # Structured data
├── job_matches/company_matches.csv       # Matches (if enabled)
└── company_workflow_log.json             # Execution log
```

## Options

- `--sitemap-url` - URL of sitemap.xml (required)
- `--company-name` - Company name for file naming (required)
- `--output-dir` - Custom output directory (default: workflow_output)
- `--resume-path` - Resume JSON for matching (optional)
- `--skip-matching` - Don't run job matching step

## More Info

- Full documentation: `src/WORKFLOW_AUTOMATION_README.md`
- Usage examples: `src/workflow_examples.sh`
- Script location: `src/workflow_automation.py`

## Troubleshooting

**Missing dependencies?**
```bash
pip install crawl4ai google-generativeai cloudscraper scikit-learn pandas numpy tqdm aiohttp
```

**API key not set?**
```bash
export GEMINI_API_KEY="your-key-here"
```

**Want to see help?**
```bash
python3 workflow_automation.py --help
```
