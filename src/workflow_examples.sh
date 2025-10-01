#!/bin/bash
# Example script demonstrating the workflow automation

# Set environment variables for testing
export GEMINI_API_KEY="AIzaSyC0HG6WUm5QdVIidnGcwl4lwK2MH2zugJo"
export GEMINI_MODEL="gemini-flash-lite-latest"

echo "========================================="
echo "Workflow Automation - Example Usage"
echo "========================================="
echo ""

# Example 1: Basic usage with skip-matching
echo "Example 1: Scraping Netflix jobs (no matching)"
echo "-----------------------------------------------"
echo "Command:"
echo "python3 workflow_automation.py \\"
echo "  --sitemap-url \"https://explore.jobs.netflix.net/careers/sitemap.xml\" \\"
echo "  --company-name \"netflix\" \\"
echo "  --skip-matching"
echo ""
echo "This will:"
echo "  1. Parse the sitemap to extract job URLs"
echo "  2. Scrape all job postings"
echo "  3. Process with LLM to get structured JSON"
echo "  4. Skip matching (no resume needed)"
echo ""

# Example 2: Full workflow with matching
echo ""
echo "Example 2: Full workflow with job matching"
echo "-------------------------------------------"
echo "Command:"
echo "python3 workflow_automation.py \\"
echo "  --sitemap-url \"https://careers.google.com/jobs/sitemap\" \\"
echo "  --company-name \"google\" \\"
echo "  --resume-path \"../runtime_data/processed_resumes/jtarriela_resume.json\""
echo ""
echo "This will:"
echo "  1. Parse the sitemap to extract job URLs"
echo "  2. Scrape all job postings"
echo "  3. Process with LLM to get structured JSON"
echo "  4. Match jobs against your resume using cosine similarity"
echo ""

# Example 3: Custom output directory
echo ""
echo "Example 3: Custom output directory"
echo "-----------------------------------"
echo "Command:"
echo "python3 workflow_automation.py \\"
echo "  --sitemap-url \"https://jobs.apple.com/sitemap/sitemap-jobs-en-us.xml\" \\"
echo "  --company-name \"apple\" \\"
echo "  --output-dir \"./apple_jobs_$(date +%Y%m%d)\" \\"
echo "  --skip-matching"
echo ""

# Example 4: Multiple companies
echo ""
echo "Example 4: Process multiple companies (using a loop)"
echo "-----------------------------------------------------"
cat << 'SCRIPT'
#!/bin/bash
companies=(
  "google|https://careers.google.com/jobs/sitemap"
  "apple|https://jobs.apple.com/sitemap/sitemap-jobs-en-us.xml"
  "netflix|https://explore.jobs.netflix.net/careers/sitemap.xml"
)

for entry in "${companies[@]}"; do
  IFS="|" read -r name url <<< "$entry"
  echo "Processing $name..."
  python3 workflow_automation.py \
    --sitemap-url "$url" \
    --company-name "$name" \
    --skip-matching
done
SCRIPT

echo ""
echo "========================================="
echo "To run an example:"
echo "  1. Make sure dependencies are installed"
echo "  2. Set environment variables:"
echo "     export GEMINI_API_KEY=\"your-key\""
echo "     export GEMINI_MODEL=\"gemini-flash-lite-latest\""
echo "  3. Run: python3 workflow_automation.py [options]"
echo "========================================="
