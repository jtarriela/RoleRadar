#!/bin/bash
# Demo script showing YAML-based CLI capabilities
# This demonstrates how configuration parameters are used from YAML

echo "=========================================="
echo "RoleRadar YAML-Based CLI Demo"
echo "=========================================="
echo ""

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "Creating config.yaml from example..."
    cp config.example.yaml config.yaml
    echo "✓ config.yaml created"
else
    echo "✓ config.yaml already exists"
fi

echo ""
echo "Configuration Overview:"
echo "----------------------"

# Show key configuration values
python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('Scraper Settings:')
print(f\"  - Max concurrent requests: {config['scraper']['max_concurrent']}\")
print(f\"  - Delay between requests: {config['scraper']['delay_between_requests']}s\")
print(f\"  - Max URLs to scrape: {config['scraper']['max_urls']}\")
print()
print('LLM Settings:')
print(f\"  - Provider: {config['llm']['provider']}\")
print(f\"  - Gemini model: {config['llm']['gemini_model']}\")
print()
print('Matcher Settings:')
print(f\"  - Minimum score filter: {config['matcher']['min_score_filter']}%\")
print(f\"  - Batch size: {config['matcher']['batch_size']}\")
print()
print('Companies configured:')
for company in config.get('companies', []):
    print(f\"  - {company['name']}: {company.get('sitemap_url', 'N/A')}\")
"

echo ""
echo "=========================================="
echo "Available CLI Commands"
echo "=========================================="
echo ""

echo "1. Show help:"
echo "   python cli.py --help"
echo ""

echo "2. Scrape jobs for a company (defined in config):"
echo "   python cli.py scrape --company meta"
echo ""

echo "3. Process scraped jobs to structured JSON:"
echo "   python cli.py process --input job_data/meta.json"
echo ""

echo "4. Parse a resume:"
echo "   python cli.py parse-resume --file resume.pdf"
echo ""

echo "5. Match jobs to resume:"
echo "   python cli.py match"
echo ""

echo "6. Run complete pipeline:"
echo "   python cli.py run-all --company meta --resume resume.pdf"
echo ""

echo "=========================================="
echo "Configuration Customization Examples"
echo "=========================================="
echo ""

echo "To adjust scraping speed, edit config.yaml:"
echo ""
echo "  scraper:"
echo "    max_concurrent: 10        # Increase for faster scraping"
echo "    delay_between_requests: 0.5  # Decrease for faster (but more aggressive)"
echo ""

echo "To change LLM provider, edit config.yaml:"
echo ""
echo "  llm:"
echo "    provider: openai          # Switch from gemini to openai"
echo "    openai_model: gpt-4o      # Use GPT-4"
echo ""

echo "To test with fewer jobs, edit config.yaml:"
echo ""
echo "  scraper:"
echo "    max_urls: 50              # Only scrape first 50 jobs"
echo ""
echo "  matcher:"
echo "    max_jobs: 50              # Only match first 50 jobs"
echo ""

echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Set your API key:"
echo "   export GEMINI_API_KEY='your-key-here'"
echo ""
echo "2. Customize config.yaml for your needs"
echo ""
echo "3. Run the pipeline:"
echo "   python cli.py run-all --company meta --resume resume.pdf"
echo ""
echo "4. Check the output:"
echo "   ls -lh runtime_data/match_results/"
echo ""
echo "For detailed documentation, see CLI_USAGE_GUIDE.md"
echo ""
