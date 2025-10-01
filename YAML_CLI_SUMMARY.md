# YAML-Based CLI Implementation Summary

This document summarizes the YAML-based CLI implementation for RoleRadar.

## What Changed

The RoleRadar project now has a YAML configuration-driven CLI that allows users to:
- Define all parameters in a single YAML file
- Run the complete job scraping and matching pipeline
- Fine-tune and tweak settings without modifying code

## Problem Statement (Addressed)

> "change the cli to take yaml instead and create yaml sections for all parameters 
> of original main scripts in src so that it can be fine tuned and tweaked so the 
> user can put the url to the xml in the yaml and then parameteriize things like 
> batch size, which model to use tec"

✅ **Solved**: Users can now configure URLs, batch sizes, models, and all other parameters in `config.yaml`

## Files Created/Modified

### New Files
1. **`cli.py`** (755 lines)
   - Main CLI entry point
   - YAML config loader
   - Commands: scrape, process, parse-resume, match, run-all

2. **`config.yaml`** (60 lines)
   - Default configuration file
   - Ready to use with minimal customization

3. **`config.example.yaml`** (92 lines)
   - Fully documented configuration template
   - Shows all available options with comments

4. **`CLI_USAGE_GUIDE.md`** (430 lines)
   - Comprehensive usage documentation
   - Examples for every command
   - Troubleshooting guide
   - Best practices

5. **`QUICK_REFERENCE.md`** (92 lines)
   - Quick reference card
   - Common commands and config tweaks
   - Compact format for quick lookup

6. **`demo_cli.sh`** (100 lines)
   - Interactive demo script
   - Shows configuration in action
   - Displays available commands

### Modified Files
1. **`README.md`**
   - Added YAML CLI section at top
   - Updated CLI commands section
   - Added links to new documentation

## Configuration Sections

All parameters from original scripts are now in YAML:

### 1. Companies
```yaml
companies:
  - name: meta
    sitemap_url: https://www.metacareers.com/sitemap.xml
```

### 2. Scraper (from raw_job_scraper.py)
```yaml
scraper:
  max_concurrent: 6
  delay_between_requests: 1.0
  max_urls: 10000
  bot_retry_delays: [10.0, 30.0, 60.0, 120.0]
  # ... 10+ parameters
```

### 3. Job Processor (from jobdata_markdown_to_json.py)
```yaml
job_processor:
  batch_size: 10
  delay_between_requests: 1.0
  max_retries: 2
  # ... 5+ parameters
```

### 4. Resume Parser (from resume_parser.py)
```yaml
resume_parser:
  use_llm: true
  output_dir: processed_resumes
```

### 5. Matcher (from cosine_similiarity.py)
```yaml
matcher:
  min_score_filter: 45.0
  batch_size: 50
  max_jobs: null
  # ... 8+ parameters
```

### 6. LLM Provider (from llm_providers.py)
```yaml
llm:
  provider: gemini
  gemini_model: gemini-1.5-flash-latest
  openai_model: gpt-4o-mini
```

### 7. Output Directories
```yaml
output:
  base_dir: runtime_data
  scraped_dir: runtime_data/scraped_jobs
  # ... 6 directories
```

## CLI Commands

### Basic Usage
```bash
python cli.py --config config.yaml COMMAND [OPTIONS]
```

### Available Commands
1. **scrape** - Scrape job postings
2. **process** - Process scraped jobs to JSON
3. **parse-resume** - Parse resume to structured JSON
4. **match** - Match jobs to resume
5. **run-all** - Run complete pipeline

## Key Features

✅ **All parameters configurable** - No code changes needed
✅ **URL management in YAML** - Sitemap URLs in config
✅ **Model selection** - Switch between Gemini/OpenAI
✅ **Batch size tuning** - Adjust for performance/cost
✅ **Fine-grained control** - 40+ configurable parameters
✅ **Flexible workflows** - Run complete pipeline or individual steps
✅ **Well documented** - 3 documentation files + examples

## Example Workflow

```bash
# 1. Setup
cp config.example.yaml config.yaml
export GEMINI_API_KEY="your-key"

# 2. Customize config.yaml
# Edit: company URLs, batch sizes, model selection

# 3. Run pipeline
python cli.py run-all --company meta --resume resume.pdf

# 4. Get results
open runtime_data/match_results/job_matches.csv
```

## Testing Performed

✅ Config loading and validation
✅ All CLI commands functional
✅ Directory creation
✅ Parameter extraction from YAML
✅ Help text generation
✅ Demo script execution

## Documentation Structure

```
RoleRadar/
├── cli.py                    # Main CLI implementation
├── config.yaml               # Default config
├── config.example.yaml       # Documented template
├── README.md                 # Project overview + quick start
├── CLI_USAGE_GUIDE.md        # Complete CLI documentation
├── QUICK_REFERENCE.md        # Quick reference card
├── demo_cli.sh              # Interactive demo
└── YAML_CLI_SUMMARY.md       # This file
```

## Migration from Old Workflow

**Old way** (workflow_automation.py):
```bash
python src/workflow_automation.py \
  --sitemap-url "https://example.com/sitemap.xml" \
  --company-name "example"
```

**New way** (YAML CLI):
```yaml
# config.yaml
companies:
  - name: example
    sitemap_url: https://example.com/sitemap.xml
```
```bash
python cli.py run-all --company example
```

## Benefits

1. **Version Control** - Config files can be committed (without API keys)
2. **Reproducibility** - Same config = same results
3. **Easy Tweaking** - Change parameters without touching code
4. **Multiple Profiles** - Different configs for test/prod
5. **Better Documentation** - YAML is self-documenting
6. **Reduced Errors** - Type checking and validation

## Next Steps for Users

1. Read `QUICK_REFERENCE.md` for quick start
2. Review `config.example.yaml` for all options
3. Consult `CLI_USAGE_GUIDE.md` for detailed examples
4. Run `demo_cli.sh` to see it in action

## Implementation Notes

- **Backward compatible** - Old scripts in `src/` still work
- **No new dependencies** - Uses existing PyYAML
- **Tested** - All commands verified working
- **Documented** - 3 levels of documentation (quick ref, guide, inline)
