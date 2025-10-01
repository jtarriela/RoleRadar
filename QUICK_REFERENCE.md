# YAML CLI Quick Reference

## Setup (One Time)

```bash
# 1. Copy example config
cp config.example.yaml config.yaml

# 2. Set API key
export GEMINI_API_KEY="your-key-here"

# 3. Edit config.yaml to customize
nano config.yaml  # or vim, code, etc.
```

## Common Commands

```bash
# Show help
python cli.py --help

# Scrape jobs for a company
python cli.py scrape --company meta

# Scrape from URL file
python cli.py scrape --url-file job_urls.txt

# Process scraped jobs
python cli.py process --input job_data/meta.json

# Parse resume
python cli.py parse-resume --file resume.pdf

# Match jobs to resume
python cli.py match

# Run everything
python cli.py run-all --company meta --resume resume.pdf
```

## Quick Config Tweaks

### Test with fewer jobs
```yaml
scraper:
  max_urls: 50  # Only scrape 50 jobs

matcher:
  max_jobs: 50  # Only match 50 jobs
```

### Speed up scraping (more aggressive)
```yaml
scraper:
  max_concurrent: 10
  delay_between_requests: 0.5
```

### Slow down scraping (safer)
```yaml
scraper:
  max_concurrent: 3
  delay_between_requests: 2.0
```

### Switch to OpenAI
```yaml
llm:
  provider: openai
  openai_model: gpt-4o-mini
```

### Adjust match threshold
```yaml
matcher:
  min_score_filter: 60.0  # Higher = fewer but better matches
```

## File Locations (default)

- **Config**: `config.yaml`
- **Scraped jobs**: `job_data/<company>.json`
- **Processed jobs**: `processed_jobs/processed_<company>.json`
- **Parsed resume**: `processed_resumes/resume.json`
- **Match results**: `runtime_data/match_results/job_matches.csv`

## Workflow Steps

1. **Scrape** → Get job postings as markdown
2. **Process** → Extract structured fields with LLM
3. **Parse Resume** → Extract resume fields with LLM
4. **Match** → Calculate similarity scores

## Common Issues

**"Configuration file not found"**
→ Run `cp config.example.yaml config.yaml`

**"API key not set"**
→ Run `export GEMINI_API_KEY="your-key"`

**"Rate limit exceeded"**
→ Increase `delay_between_requests` in config

**"Bot detection"**
→ Decrease `max_concurrent` and increase `delay_between_requests`

## Full Documentation

- Complete guide: [`CLI_USAGE_GUIDE.md`](CLI_USAGE_GUIDE.md)
- Config reference: [`config.example.yaml`](config.example.yaml)
- Project README: [`README.md`](README.md)
