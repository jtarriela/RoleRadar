import os
import json
import asyncio
from typing import List, Dict, Any
# --- NEW: hyperlink stripping helper (drop-in replacement) -------------------
import re

URL_LINE_RE = re.compile(r"""
(
    \[[^\]]+\]\([^)]+\)                # [text](url)
  | <https?://[^>\s]+>                 # <https://...>
  | https?://\S+                       # raw http(s)
  | \b(?:www\.)?[a-z0-9][\w.-]+\.(?:com|org|net|io|co|edu|gov)(?:/\S*)?
)
""", re.IGNORECASE | re.VERBOSE)

def drop_hyperlink_lines(md: str, max_chars: int | None = None) -> str:
    """Remove any entire line that contains a hyperlink; optionally cap to max_chars."""
    out = []
    for line in md.splitlines():
        if URL_LINE_RE.search(line):
            continue
        out.append(line)
    cleaned = "\n".join(out)
    return cleaned[:max_chars] if isinstance(max_chars, int) and max_chars > 0 else cleaned
# -----------------------------------------------------------------------------
# --- SAFE IMPORTS ---
import google.generativeai as genai
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# --- 1. CONFIGURATION ---
try:
    API_KEY = "AIzaSyBRLSOoung9KHxvWcxbrEEhgCLNBio3bF4"
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    exit()

COMPREHENSIVE_JOB_KEYWORDS = [
    "job", "career", "opening", "position", "role", "opportunity", "listing", "details",
    "summary", "description", "overview", "duties", "responsibilities", "accountabilities",
    "qualifications", "requirements", "experience", "skills", "background", "profile",
    "education", "degree", "preferred", "minimum", "basic", "engineer", "developer",
    "software", "hardware", "analyst", "data", "scientist", "manager", "specialist",
    "architect", "designer", "researcher", "lead", "senior", "principal", "staff",
    "full-time", "part-time", "contract", "internship", "temporary", "permanent",
    "benefits", "salary", "compensation", "location", "remote", "onsite", "hybrid",
    "team", "culture", "growth", "development", "technology", "tools", "platform",
    "project", "innovation", "equal opportunity", "diversity", "inclusion", "apply",
    "application", "hiring", "recruitment", "eeo"
]

# --- 2. THE AGENT'S "BRAINS" - SPECIALIZED LLM FUNCTIONS ---

async def is_job_description_page(markdown_content: str, url: str) -> bool:
    """
    The "Triage Brain" with an improved, more flexible prompt.
    """
    print(f"  [AI Triage] Analyzing content from {url}...")
    
    # --- THIS IS THE NEW DEBUGGING CODE ---
    # It will save the markdown to a file so you can inspect it.
    if not os.path.exists("markdown_logs"):
        os.makedirs("markdown_logs")
    
    # Create a safe filename from the URL
    filename = url.replace("https://", "").replace("http://", "").replace("/", "_") + ".md"
    filepath = os.path.join("markdown_logs", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# MARKDOWN FOR URL: {url}\n\n")
        f.write(markdown_content)
    
    print(f"    [DEBUG] Saved markdown for this URL to: {filepath}")
    # --- END OF DEBUGGING CODE ---
    
    # --- NEW: strip hyperlink lines + cap size before LLM --------------------
    triage_md = drop_hyperlink_lines(markdown_content, max_chars=30000)

    prompt = f"""
    You are a page classifier. Your task is to determine if the following Markdown content represents a single, specific job description.

    A job description page describes ONE role, not a list of many.
    Look for common sections like a job title, a summary, and a list of qualifications or responsibilities. Be flexible with the exact wording.
    Return your answer as a single, clean JSON object: {{"is_job_description": true}} or {{"is_job_description": false}}

    Markdown Content (first 8000 chars):
    ---
    {triage_md} 
    ---
    """
    try:
        response = await model.generate_content_async(prompt)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        analysis = json.loads(cleaned_text)
        return analysis.get("is_job_description", False)
    except Exception as e:
        print(f"  [LLM ERROR] Triage failed: {e}")
        return False
    
async def extract_job_data_with_llm(markdown_content: str, url: str) -> Dict[str, Any]:
    """The "Extractor Brain" - remains highly effective."""
    print(f"    [AI Extractor] Extracting structured data from: {url}")
    
    # --- NEW: strip hyperlink lines + cap size --------------------------------
    extract_md = drop_hyperlink_lines(markdown_content, max_chars=25000)
    # --------------------------------------------------------------------------
    
    prompt = f"""
    You are an expert data extractor. The following Markdown is from a job description page.
    Read the content and extract the information into a clean JSON object with the schema:
    {{
      "job_title": "...",
      "location": "...",
      "summary": "...",
      "qualifications": [...]
      "Pay_range": "..."
      ""benefits": "..."
    }}

    Markdown Content:
    ---
    {extract_md}
    ---
    """
    try:
        response = await model.generate_content_async(prompt)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        extracted_data = json.loads(cleaned_text)
        extracted_data["url"] = url

        # Optional: keep the full body you sent to the LLM so you always have it
        # extracted_data.setdefault("description_full", extract_md)

        return extracted_data
    except Exception as e:
        print(f"  [LLM ERROR] Extraction failed: {e}")
        return {}

# --- 3. MAIN WORKFLOW EXECUTION ---
async def main():
    start_url = "https://jobs.apple.com/en-us/search"
    all_found_jobs = []

    print(f"--- STARTING ADVANCED CRAWL AT: {start_url} ---\n")

    # Move the just-in-time imports here to avoid the circular import bug
    from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
    from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

    filter_chain = FilterChain([
        URLPatternFilter(
            patterns=["*jobs.apple.com/en-us/search*", "*jobs.apple.com/en-us/details/*"],
        )
    ])
    
    keyword_scorer = KeywordRelevanceScorer(keywords=COMPREHENSIVE_JOB_KEYWORDS, weight=0.8)

    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=1,
            include_external=False,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer,
            max_pages=10
        ),
        stream=True,
        verbose=False
    )

    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(start_url, config=config):
            if result.url == start_url:
                print(f"[CRAWLER] Starting analysis from root page: {result.url}")
                continue

            score = result.metadata.get("score", 0)
            print(f"\n[CRAWLER] Processing result | Score: {score:.2f} | URL: {result.url}")

            if not result.markdown or not result.markdown.raw_markdown.strip():
                print("  [WARN] Page had no Markdown content to analyze. Skipping.")
                continue

            is_job_page = await is_job_description_page(result.markdown.raw_markdown, result.url)
            
            if is_job_page:
                print(f"  [AI Triage] Verdict: YES, this is a job page.")
                job_data = await extract_job_data_with_llm(result.markdown.raw_markdown, result.url)
                if job_data:
                    all_found_jobs.append(job_data)
            else:
                print(f"  [AI Triage] Verdict: NO, skipping this page.")

    print(f"\n--- WORKFLOW COMPLETE ---")
    print(f"Analyzed high-value pages and successfully extracted {len(all_found_jobs)} jobs.")
    
    with open("apple_jobs_debug.json", "w", encoding="utf-8") as f:
        json.dump(all_found_jobs, f, indent=2, ensure_ascii=False)
    
    print("Results saved to apple_jobs_debug.json")

if __name__ == "__main__":
    asyncio.run(main())