# RoleRadar Pipeline

RoleRadar is now driven by a single YAML-controlled pipeline that crawls job
sites, cleans and parses postings, processes a résumé, and produces detailed
match reports. The legacy multi-command CLI has been retired; every stage is
coordinated by `run_pipeline.py` so that one configuration file captures the
entire run.

## What's Included

- **Sitemap discovery** (optional): fetch and filter sitemap XML to produce raw
  job URLs.
- **Markdown scraping**: pull job pages with concurrency controls and
  anti-bot detection.
- **LLM job parsing**: convert scraped markdown into structured JSON using the
  configured LLM provider.
- **Résumé parsing**: run an LLM-first parser with a regex fallback for
  structured résumé JSON.
- **Structured matching**: combine rule-based checks with embeddings to produce
  ranked matches, highlight gaps, and generate improvement suggestions.
- **Run artefacts**: every intermediate dataset plus the final results are
  written to a timestamped directory for easy inspection and debugging.

## Quick Start

1. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If you maintain your own dependency list, ensure the environment provides:
   `aiohttp`, `crawl4ai`, `cloudscraper`, `python-dotenv`, `pyyaml`,
   `sentence-transformers`, `scikit-learn`, `pandas`, `tqdm`, and your chosen
   LLM SDKs (e.g. `openai` or `google-generativeai`).

3. **Create a configuration file**
   Copy the template and tailor it to your run:
   ```bash
   cp config/pipeline.example.yaml config/pipeline.local.yaml
   ```
   Update API keys under `secrets`, point `resume.path` to your résumé, and
   define the sites you want to crawl.

4. **Run the pipeline**
   ```bash
   python run_pipeline.py --config config/pipeline.local.yaml
   ```

The script prints progress as it moves through sitemap extraction, scraping,
parsing, résumé processing, and matching. All artefacts are stored in
`runtime_data/pipeline_runs/<run-id>/` unless you override `runtime.output_dir`.
Check the generated `run_summary.json` for direct paths to every output.

## Configuration Reference

Key sections of the YAML file:

- `runtime`: choose an output directory and (optionally) a fixed `run_id`.
- `logging`: control verbosity and per-run log filename.
- `secrets`: inline environment overrides such as `OPENAI_API_KEY` or
  `GOOGLE_API_KEY`. Leave values empty to rely on `.env` or shell exports.
- `resume`: point to the résumé to parse and choose whether to use the LLM
  parser (`use_llm: true|false`).
- `sites`: list of site definitions. Each supports:
  - `urls`: direct lists, existing files, or sitemap URLs.
  - `url_filters`: include/exclude patterns or domain allowlists.
  - `extractor`: optional sitemap extraction tuning (delays, proxies, user
    agents).
  - `scraper`: choose a built-in scraper via `company_key` or provide include /
    exclude patterns for the generic pattern scraper. Concurrency, delay, and
    retry settings are configurable per site.
  - `parser`: controls for the async LLM processor (batch size, concurrency,
    retry schedule, output filename).
- `matching`: enable/disable matching, specify embedding model, tweak batch
  size, minimum score cutoff, and output filenames.

Refer to `config/pipeline.example.yaml` for a complete annotated template.

## Output Structure

Each pipeline run produces a directory similar to:

```
runtime_data/pipeline_runs/20240102-235959/
  pipeline.log                # combined log output
  resume.json                 # parsed résumé
  run_summary.json            # overview of every artefact
  sites/
    capitalone/
      sitemaps/...
      scraped/capitalone_jobs_...json
      parsed/capitalone_jobs.json
    custom_example/
      ...
  matches/
    matches.json              # JSON array of structured matches
    matches_detailed.csv      # CSV with full breakdown
```

`run_summary.json` captures the important counts (URLs processed, jobs parsed,
matching success rate) and the paths to every generated file.

## Pipeline Internals

- **`src/raw_job_scraper.py`**: provides company-specific scrapers and the
  pattern-based scraper used for custom sites. Handles bot detection, retries,
  markdown cleaning, and checkpointing.
- **`src/jobdata_markdown_json_concurrent.py`**: asynchronous LLM processing
  with configurable concurrency, retry backoff, and detailed processing stats.
- **`src/resume_parser.py`**: LLM-first résumé parser with a deterministic
  fallback, plus helpers for saving structured JSON.
- **`src/structured_text_mapping.py`**: structured matching logic combining
  experience, skills, education, and semantic embedding scores.
- **`run_pipeline.py`**: orchestration layer that wires every stage together and
  writes the run summary.

## Working With Secrets

You can either define secrets under the `secrets` section of the YAML file
(values are exported to the environment at runtime) or rely on a local `.env`
that contains the same keys. The pipeline loads `.env` automatically via
`python-dotenv` before applying YAML overrides.

## Troubleshooting

- **Missing packages**: confirm your environment matches the dependency list
  above.
- **Rate limit errors**: lower `max_concurrent_requests` in the parser settings
  or adjust site-specific scraper concurrency.
- **Empty outputs**: inspect `run_summary.json` and the per-stage logs under the
  relevant site directory to identify where URLs were filtered or parsing
  failed.

## Next Steps

The current pipeline produces machine-friendly artefacts ready for database
integration or a future front-end. Natural follow-ons include:

1. Persisting parsed jobs, résumé snapshots, and match results into a SQL store
   (e.g. Postgres + pgvector) keyed by `run_id`.
2. Exposing a lightweight service layer for browsing run history and re-running
   individual stages.
3. Building a web UI or analytics notebook workflow on top of that service for
   richer exploration and visualisation of matches.

---

With the CLI removed, `run_pipeline.py` is the single entrypoint for RoleRadar
runs. Tune your YAML, execute once, and inspect the automatically organised
outputs for every stage of the pipeline.
