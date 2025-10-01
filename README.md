## 🚀 YAML-Based CLI (NEW!)

**Configure everything in YAML!** The new CLI allows you to define all parameters in a YAML configuration file, making it easy to fine-tune and customize the entire pipeline.

```bash
# Quick start
export GEMINI_API_KEY="your-key-here"

# Copy example config and customize
cp config.example.yaml config.yaml

# Run complete pipeline
python cli.py --config config.yaml run-all --company meta --resume resume.pdf

# Or run individual steps
python cli.py --config config.yaml scrape --company meta
python cli.py --config config.yaml process --input job_data/meta.json
python cli.py --config config.yaml parse-resume --file resume.pdf
python cli.py --config config.yaml match
```

**Key Features:**
- 📝 **YAML Configuration**: All parameters in one file - scraper settings, batch size, models, output paths
- 🔧 **Fine-tuning**: Easily adjust max_concurrent, delays, retry logic, batch sizes
- 🎯 **Model Selection**: Configure which LLM provider and model to use
- 📊 **Flexible Workflows**: Run complete pipeline or individual steps
- 🔗 **URL Management**: Specify sitemap URLs or job URL files in config

**Documentation:**
- **Configuration Reference:** [`config.example.yaml`](config.example.yaml) - Fully commented example
- **Old Workflow Tool:** [`src/workflow_automation.py`](src/workflow_automation.py) - Previous implementation

---

## Getting started (quick)

These steps get you a working development environment and show common commands for the CLI.
Prerequisites
- Python 3.11+ (or your project's supported Python)
- git, pip

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) (Optional) Install LLM SDKs when testing LLM parsing:

```bash
pip install openai google-generativeai
```

4) Show CLI help (a convenience script `startcli.py` is provided so you can run the CLI as `python startcli.py ...`)

```bash
python startcli.py --help
```

Quick examples

- Parse a plain-text résumé (no LLM):

```bash
python startcli.py resume parse --file /path/to/resume.txt --out outputs/resume.json --no-use-llm
```

- Normalize cached HTML to CSV (place `.html` files into `.cache` first):

```bash
python startcli.py normalize --cache .cache --out outputs/jobs.csv
```

- Match résumé to jobs and write matches CSV:

```bash
python startcli.py match --resume outputs/resume.json --jobs outputs/jobs.csv --out outputs/matches.csv
```

- Generate a text report from matches:

```bash
python startcli.py report --matches outputs/matches.csv --limit 20
```

Testing

Run the unit tests from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

If tests fail with import errors, ensure you're running from the repo root with the virtualenv active. You can also set:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Note: one resume test is skipped unless `jtarriela_resume[sp].pdf` exists in the repo root.

Environment variables / secrets

Create a local, gitignored `.env` or export keys directly in your shell. Typical environment variables used by the project are:

```
OPENAI_API_KEY=
GOOGLE_API_KEY=
GEMINI_MODEL=
LLM_PROVIDER=
```

Developer notes

- `startcli.py` is a thin wrapper so you can run `python startcli.py ...` in development.
- Tests and code import the `jobflow` package directly. There used to be a temporary `RoleRadar/` shim during development; the codebase now imports `jobflow.*`.
- `jobflow/ingest/lever_adapter.py` currently contains a placeholder implementation — implement it or patch adapters in tests to run full integration flows.
- Consider adding a `pyproject.toml` / `setup.cfg` and a console script entrypoint if you want `pip install -e .` to install a `jobflow` command.

If you'd like, I can add `.env.example`, a demo `tests/data/` file, or implement the Lever adapter next — tell me which and I'll add it.
# RoleRadar
# MVP Goal (CLI)

Ship a local tool that:

1. crawls target companies’ careers pages (Scrythe for hard sites; GH/Lever adapters for easy sites),
2. normalizes raw HTML → `jobs.csv`,
3. parses a resume → `resume.json` + embedding,
4. ranks matches → `matches.csv` with reasons.

No web UI, no DB.

---

# Scope / Non-Goals

**In**: CLI, local cache, CSV outputs, GH/Lever adapters, Scrythe runner, LLM resume parser, basic ranker (filters → vector → LLM-judge).
**Out**: user accounts, auto-apply, dashboards, Postgres/pgvector (later), scheduling/cron (optional bonus).

---

# CLI Commands

## New YAML-Based CLI (Recommended)

All configuration in YAML - easy to customize and version control:

```bash
# Show available commands
python cli.py --help

# Scrape jobs for a company (defined in config.yaml)
python cli.py --config config.yaml scrape --company meta

# Scrape jobs from URL file
python cli.py --config config.yaml scrape --url-file job_urls.txt

# Process scraped markdown to structured JSON
python cli.py --config config.yaml process --input job_data/meta.json

# Parse resume to structured JSON
python cli.py --config config.yaml parse-resume --file resume.pdf

# Match jobs to resume using embeddings
python cli.py --config config.yaml match

# Run complete pipeline
python cli.py --config config.yaml run-all --company meta --resume resume.pdf
```

**Configuration File (config.yaml):**
```yaml
# Configure scraper
scraper:
  max_concurrent: 6              # Adjust concurrency
  delay_between_requests: 1.0    # Rate limiting
  max_urls: 10000                # Limit for testing
  batch_size: 50                 # Jobs per batch

# Configure LLM
llm:
  provider: gemini               # or 'openai'
  gemini_model: gemini-1.5-flash-latest

# Configure matching
matcher:
  min_score_filter: 45.0         # Minimum match score
  batch_size: 50                 # Embedding batch size
```

See [`config.example.yaml`](config.example.yaml) for complete configuration reference.

## Legacy CLI Commands (v0)

For reference - the original planned interface:

```
jobflow resume parse <resume_path> --out ./resume.json
jobflow crawl --companies meta google nvidia --cache ./.cache
jobflow normalize --cache ./.cache --out ./outputs/jobs.csv
jobflow match --resume ./resume.json --jobs ./outputs/jobs.csv \
              --location "NYC, Remote-US" --min-pay 160000 \
              --threshold 0.65 --topk 50 \
              --out ./outputs/matches.csv
jobflow report --matches ./outputs/matches.csv --limit 20
```

---

# Repo Layout

```
jobflow/
  cli.py
  config.yaml
  extern/scrythe/               # submodule or sibling repo
  collect/runner.py             # wraps scrythe, logs sessions
  normalize/
    schema.py                   # JobRow dataclass + headers
    html_to_fields.py           # rules + LLM fallback extractor
    write_csv.py
  ingest/
    greenhouse_adapter.py
    lever_adapter.py
  resume/
    parse_resume.py             # LLM → JSON
    embed.py                    # build embeddings
  rank/
    prefilter.py
    vector_rank.py
    llm_judge.py
    aggregate.py
  outputs/                      # jobs.csv, matches.csv
  .env.example
  requirements.txt
  README.md
```

---

# Config (example)

```yaml
companies:
  - name: meta
    careers_url: https://www.metacareers.com/jobs
    portal_hint: workday|custom
  - name: openai
    greenhouse_token: openai
  - name: databricks
    lever_slug: databricks

crawl:
  cache_dir: ./.cache
  max_tabs_per_domain: 1
  delay_ms_range: [500, 2000]
  daily_page_cap_per_domain: 150
  use_residential_proxies: false

matching:
  threshold: 0.65
  topk_llm: 30
  weights:
    prefilter: 0.30
    vector:    0.45
    llm:       0.25
```

---

# Data Schemas (CSV/JSON)

**`jobs.csv` (headers)**

```
job_id,company,source,ext_id,title,department,employment_type,locations,
remote_flag,onsite_flag,pay_min,pay_max,pay_currency,posted_date,updated_at,
absolute_url,description_text,description_html,scraped_at,source_fingerprint
```

**`resume.json`**

```json
{
  "full_name": "...",
  "contact": {"email":"...","phone":"..."},
  "roles": [{"title":"Sr HPC Engineer","years":4,"skills":["CUDA","MPI","CTH"]}],
  "skills": ["CUDA","HPC","Python","C++"],
  "education": [{"degree":"MS","field":"ME","year":2020}],
  "yoe_total": 8,
  "preferences": {"locations":["NYC","Remote-US"],"min_pay":160000,"remote_ok":true},
  "embedding": [/* vector */]
}
```

---

# Core Prompts

**Resume → JSON (strict)**

```
Return ONLY valid JSON:
{full_name, contact:{email,phone}, roles:[{title,years,skills[]}],
 skills[], education:[{degree,field,year}], yoe_total, preferences?}
- Infer years per role and total YOE (integer).
- Canonicalize skills (dedupe, standard names).
```

**Job HTML fallback → fields**

```
Extract JSON:
{company,title,department,employment_type,locations[],pay_min,pay_max,pay_currency,
 posted_date,updated_at,absolute_url,description_text,description_html,source}
- If a field is missing, return null (not "N/A").
- locations: array; include 'Remote' if applicable.
```

**LLM judge (resume↔job)**

```
Given RESUME_JSON and JOB_TEXT, return:
{relevance: 0..1, reasons: [2-4 bullets], must_have_gaps: [strings]}
- Weight hard-skill overlap & seniority match.
- Penalize missing "must-have" keywords.
```

---

# Anti-Bot Posture (for crawl)

* Concurrency cap per domain (1–2), randomized dwell/scroll, exponential backoff, daily page caps, optional proxies.
* Idempotent cache: skip pages already fetched today.
* Session metrics: pages_fetched, retries, blocks, challenges.

---

# Telemetry / Logs (MVP)

* `collect.log`: per-run stats (company, pages, blocks, duration).
* `normalize.log`: files processed, rows written, duplicates skipped.
* `match.log`: N jobs filtered, M vector-ranked, K LLM-judged.

---

# Acceptance Criteria

* **Crawl**: At least 3 target companies; ≥90% of listings cached without manual fixes.
* **Normalize**: `jobs.csv` with ≥95% rows containing `company,title,absolute_url,description_text`.
* **Resume parse**: stable JSON with skills & YOE on 3+ sample resumes.
* **Match**: `matches.csv` with top-K jobs; each has `fit_score (0..1)` and 2–4 “why” bullets.
* **Idempotency**: Re-running crawl+normalize does not duplicate rows (fingerprint check).

---

# Test Plan (CLI)

1. Unit: schema serialization, fingerprinting, HTML → fields rules, cosine similarity.
2. Integration: crawl → normalize → match on 2 GH/Lever companies + 1 Workday/custom.
3. Smoke: corrupted HTML is skipped, not fatal; missing pay is null, not “0”.

---

# Security & Compliance (MVP)

* Respect site ToS and robots where required; per-domain caps; stop on challenge spikes.
* Keep raw HTML local; don’t redistribute.
* Store API keys in `.env`; never commit.

---

# Next Steps (post-MVP)

* Add SQLite (saved searches), then Postgres + pgvector.
* Add Workday/SuccessFactors smart adapters (network JSON capture).
* Add auto-apply for GH/Lever only.
* Optional web front-end (reads `jobs.csv`/`matches.csv` initially).

---

If you want, I’ll drop starter files for `cli.py`, `schema.py`, and the GH/Lever adapters exactly matching these specs.
RoleRadar
Overview

RoleRadar is a lightweight command‑line tool that helps candidates match their
résumés against open positions at target companies. It scrapes job postings
from careers pages, normalises them into a CSV, parses a candidate résumé
into structured JSON, computes vector similarities and uses an LLM to
produce relevance scores and qualitative reasons. The final matches can
be exported as CSV or summarised in a human‑readable report.

Key Features

Resume parsing via LLM – Résumé parsing now relies primarily on a
large language model. By default the system will use a Gemini provider
(gemini‑1.5‑pro) to convert unstructured résumé text into a
structured JSON schema. When no API key is configured, a simple
heuristic parser extracts names, contact information, roles, skills and
education.

Crawling and normalisation – Fetch careers pages via Scrythe or
platform adapters, normalising raw HTML into a well‑defined set of job
fields.

Ranking pipeline – Filter jobs based on candidate preferences,
compute cosine similarity between résumé and job embeddings, judge
relevance using an LLM and aggregate the scores into a final fit score.

CLI Usage

The jobflow command provides subcommands to run each stage of the
pipeline. Examples: