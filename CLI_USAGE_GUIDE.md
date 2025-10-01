# YAML-Based CLI Usage Guide

This guide shows you how to use the new YAML-based CLI for RoleRadar, which allows you to configure all parameters in a YAML file for easy customization and fine-tuning.

## Quick Start

1. **Copy the example configuration:**
   ```bash
   cp config.example.yaml config.yaml
   ```

2. **Set your API key:**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   # OR
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Customize your config.yaml** (see Configuration Reference below)

4. **Run the pipeline:**
   ```bash
   # Scrape jobs for a company
   python cli.py --config config.yaml scrape --company meta
   
   # Process scraped jobs
   python cli.py --config config.yaml process --input job_data/meta.json
   
   # Parse your resume
   python cli.py --config config.yaml parse-resume --file your_resume.pdf
   
   # Match jobs to resume
   python cli.py --config config.yaml match
   
   # Or run everything at once
   python cli.py --config config.yaml run-all --company meta --resume your_resume.pdf
   ```

## Configuration Reference

### Companies Section

Define companies you want to scrape. Each company needs either a sitemap URL or you can provide a text file with job URLs.

```yaml
companies:
  - name: meta
    sitemap_url: https://www.metacareers.com/sitemap.xml
  - name: netflix
    sitemap_url: https://jobs.netflix.com/sitemap.xml
```

### Scraper Configuration

Fine-tune the job scraping behavior:

```yaml
scraper:
  max_concurrent: 6                    # Number of parallel requests
  delay_between_requests: 1.0          # Seconds to wait between requests
  max_retries: 3                       # Retry failed requests
  timeout_seconds: 30                  # Request timeout
  max_urls: 10000                      # Max URLs to scrape (null = unlimited)
  
  output_dir: job_data                 # Where to save scraped markdown
  max_chars: 100000                    # Max chars per job (null = unlimited)
  
  # Bot evasion settings
  user_agent_rotation: true            # Rotate user agents
  randomize_delays: true               # Add random delays
  bot_retry_delays: [10.0, 30.0, 60.0, 120.0]  # Delays on bot detection
  max_bot_retries: 3                   # Max retries on bot detection
```

**Key parameters to adjust:**
- `max_concurrent`: Higher = faster but more likely to trigger rate limits
- `delay_between_requests`: Lower = faster but more aggressive
- `max_urls`: Set to small number (e.g., 50) for testing

### Job Processor Configuration

Configure how scraped markdown is converted to structured JSON:

```yaml
job_processor:
  batch_size: 10                       # Jobs to process before checkpoint
  delay_between_requests: 1.0          # Delay between LLM calls (rate limiting)
  max_retries: 2                       # Retry failed parsing
  output_dir: processed_jobs           # Where to save processed JSON
```

**Key parameters to adjust:**
- `batch_size`: Smaller = more frequent checkpoints, larger = fewer API overhead
- `delay_between_requests`: Increase if hitting rate limits

### Resume Parser Configuration

Configure resume parsing:

```yaml
resume_parser:
  use_llm: true                        # Use LLM for parsing (vs. simple regex)
  output_dir: processed_resumes        # Where to save parsed resumes
```

### Matcher Configuration

Configure job matching with embeddings:

```yaml
matcher:
  # File paths
  resume_path: runtime_data/processed_resumes/resume.json
  jobs_path: runtime_data/processed_job_data/processed_jobs.json
  output_path: runtime_data/match_results/job_matches.csv
  
  # Matching parameters
  max_jobs: null                       # Limit for testing (null = all jobs)
  min_score_filter: 45.0               # Minimum match score (0-100)
  batch_size: 50                       # Embedding batch size
  
  # Cost control
  confirm_high_cost: true              # Ask before running expensive operations
```

**Key parameters to adjust:**
- `max_jobs`: Set to 100 for quick testing
- `min_score_filter`: Higher = fewer but better matches
- `batch_size`: Adjust based on API rate limits

### LLM Provider Configuration

Choose and configure your LLM provider:

```yaml
llm:
  provider: gemini                     # 'gemini', 'openai', or 'placeholder'
  
  # Model selection
  gemini_model: gemini-1.5-flash-latest  # Fast and cheap
  # gemini_model: gemini-1.5-pro         # Slower but more accurate
  
  openai_model: gpt-4o-mini            # or gpt-4o, gpt-3.5-turbo
  
  requests_per_minute: 60              # Rate limiting
```

**Provider comparison:**
- **Gemini (recommended)**: Free tier available, good quality, fast
- **OpenAI**: High quality, costs money immediately
- **Placeholder**: No LLM, basic regex parsing only

### Output Directories

Configure where all outputs are saved:

```yaml
output:
  base_dir: runtime_data
  sitemap_dir: runtime_data/sitemaps
  scraped_dir: runtime_data/scraped_jobs
  processed_dir: runtime_data/processed_job_data
  resume_dir: runtime_data/processed_resumes
  matches_dir: runtime_data/match_results
```

### Workflow Configuration

Control which steps run automatically:

```yaml
workflow:
  run_sitemap_parse: true
  run_scraper: true
  run_job_processor: true
  run_resume_parser: false             # Only if resume provided
  run_matcher: false                   # Only if resume and jobs available
  
  continue_on_error: false             # Stop on first error
  save_checkpoints: true               # Save progress regularly
```

## CLI Commands

### Scrape Jobs

```bash
# Scrape from company in config
python cli.py --config config.yaml scrape --company meta

# Scrape from URL file
python cli.py --config config.yaml scrape --url-file job_urls.txt
```

The scraper will:
1. Read URLs from the sitemap or file
2. Fetch each job posting
3. Save markdown content to `job_data/<company>.json`

### Process Jobs

```bash
python cli.py --config config.yaml process --input job_data/meta.json
```

Optional output file:
```bash
python cli.py --config config.yaml process \
  --input job_data/meta.json \
  --output custom_output.json
```

The processor will:
1. Read scraped markdown
2. Use LLM to extract structured fields (title, company, salary, etc.)
3. Save to `processed_jobs/<filename>.json`

### Parse Resume

```bash
# Parse PDF resume
python cli.py --config config.yaml parse-resume --file resume.pdf

# Parse DOCX resume
python cli.py --config config.yaml parse-resume --file resume.docx

# Parse text resume
python cli.py --config config.yaml parse-resume --file resume.txt
```

Custom output:
```bash
python cli.py --config config.yaml parse-resume \
  --file resume.pdf \
  --output my_resume.json
```

The parser will:
1. Extract text from file
2. Use LLM to identify skills, experience, education
3. Save to `processed_resumes/<filename>.json`

### Match Jobs

```bash
# Use paths from config
python cli.py --config config.yaml match

# Override paths
python cli.py --config config.yaml match \
  --resume my_resume.json \
  --jobs processed_jobs/meta.json \
  --output matches.csv
```

The matcher will:
1. Generate embeddings for resume and all jobs
2. Calculate cosine similarity scores
3. Filter by minimum score threshold
4. Save ranked matches to CSV

### Run Complete Pipeline

```bash
# Scrape, process, and optionally match
python cli.py --config config.yaml run-all --company meta

# Include resume parsing and matching
python cli.py --config config.yaml run-all \
  --company meta \
  --resume resume.pdf
```

## Example Workflows

### Quick Test (50 jobs)

Edit config.yaml:
```yaml
scraper:
  max_urls: 50  # Limit to 50 jobs for quick test

matcher:
  max_jobs: 50  # Only match first 50 jobs
```

Run:
```bash
python cli.py --config config.yaml run-all --company meta --resume resume.pdf
```

### Production Run (All Jobs)

Edit config.yaml:
```yaml
scraper:
  max_urls: null  # No limit
  max_concurrent: 6  # Moderate concurrency
  delay_between_requests: 1.0  # Safe rate limiting

matcher:
  max_jobs: null  # Match all jobs
  min_score_filter: 55.0  # Higher threshold for better matches
```

Run individual steps with monitoring:
```bash
# Step 1: Scrape (may take hours for large companies)
python cli.py --config config.yaml scrape --company meta

# Step 2: Process (uses LLM API, costs money)
python cli.py --config config.yaml process --input job_data/meta.json

# Step 3: Parse resume
python cli.py --config config.yaml parse-resume --file resume.pdf

# Step 4: Match
python cli.py --config config.yaml match
```

### Multiple Companies

Edit config.yaml:
```yaml
companies:
  - name: meta
    sitemap_url: https://www.metacareers.com/sitemap.xml
  - name: google
    sitemap_url: https://careers.google.com/sitemap.xml
  - name: netflix
    sitemap_url: https://jobs.netflix.com/sitemap.xml
```

Run for each company:
```bash
for company in meta google netflix; do
  echo "Processing $company..."
  python cli.py scrape --company $company
  python cli.py process --input job_data/${company}.json
done

# Then match against all
python cli.py parse-resume --file resume.pdf
python cli.py match --jobs processed_jobs/processed_*.json
```

## Troubleshooting

### "Configuration file not found"
Make sure you've created `config.yaml`:
```bash
cp config.example.yaml config.yaml
```

### "GEMINI_API_KEY not set"
Set your API key:
```bash
export GEMINI_API_KEY="your-key-here"
```

Or add to `.env` file in the repository root.

### "Rate limit exceeded"
Increase delays in config:
```yaml
scraper:
  delay_between_requests: 2.0  # Increase delay

job_processor:
  delay_between_requests: 2.0  # Increase delay
  batch_size: 5  # Smaller batches
```

### "Bot detection" or CAPTCHAs
The scraper will automatically retry with longer delays. If it keeps failing:
```yaml
scraper:
  max_concurrent: 3  # Reduce concurrency
  delay_between_requests: 2.0  # Increase delay
  randomize_delays: true  # Ensure this is enabled
```

### Memory issues with large datasets
Process in smaller batches:
```yaml
scraper:
  max_urls: 1000  # Process 1000 at a time

matcher:
  batch_size: 25  # Smaller embedding batches
```

## Tips & Best Practices

1. **Start Small**: Always test with `max_urls: 50` before running full scrapes

2. **Save API Costs**: Use `use_llm: false` for resume parsing if you just want to test the pipeline

3. **Monitor Progress**: The CLI saves checkpoints regularly. If a run fails, you can resume from the last checkpoint

4. **Version Control**: Commit your `config.yaml` (without API keys!) to track configuration changes

5. **Multiple Configs**: Create different config files for different scenarios:
   ```bash
   python cli.py --config config.test.yaml run-all --company meta
   python cli.py --config config.prod.yaml run-all --company meta
   ```

6. **Review Results**: The match output is a CSV - open in Excel/Google Sheets to review and sort by score

## Advanced: Customizing for Your Use Case

### Adjust Matching Threshold by Job Level

Create multiple config files with different thresholds:

**config.senior.yaml**:
```yaml
matcher:
  min_score_filter: 65.0  # Higher bar for senior roles
```

**config.entry.yaml**:
```yaml
matcher:
  min_score_filter: 45.0  # Lower bar for entry roles
```

### Different Models for Different Tasks

```yaml
llm:
  provider: gemini
  gemini_model: gemini-1.5-flash-latest  # Fast for job processing
  
# For resume parsing, temporarily switch to pro:
# gemini_model: gemini-1.5-pro
```

### Custom Output Locations

```yaml
output:
  base_dir: /mnt/data/job_search  # Use external drive
  processed_dir: /mnt/data/job_search/processed
```

## Next Steps

1. Read [`config.example.yaml`](config.example.yaml) for all available options
2. Check [`README.md`](README.md) for overall project documentation
3. See [`src/`](src/) for implementation details of each component

For issues or questions, please open a GitHub issue.
