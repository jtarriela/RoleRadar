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

# CLI Commands (v0)

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
