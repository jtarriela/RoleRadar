To run the RoleRadar CLI you need a Python environment with the project unpacked and its dependencies installed. The project’s `cli.py` defines the subcommands you can call, such as `resume parse`, `crawl`, `normalize`, `match` and `report`. Here’s a step‑by‑step guide:

1. **Unpack the codebase** (if you downloaded a tarball):

   ```bash
   tar -xzf RoleRadar_updated.tar.gz
   cd RoleRadar
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes optional dependencies for OpenAI and Gemini; they’ll only be used if you provide API keys.

4. **Set your API keys**.  The application reads API keys from environment variables.  You can either export them directly:

   ```bash
   export OPENAI_API_KEY="sk‑...your‑key..."
   export GEMINI_API_KEY="your‑gemini‑key"
   export LLM_PROVIDER=openai    # optional: choose openai, gemini or placeholder
   ```

   …or copy `.env.example` to `.env`, fill in your keys, and load it into your shell:

   ```bash
   cp .env.example .env
   # edit .env with your keys
   set -a            # enable automatic export
   source .env       # load variables into environment
   set +a
   ```

5. **Run the CLI**.  From the repository root you can invoke the CLI using Python’s module syntax:

   ```bash
   python -m jobflow.cli resume parse --file my_resume.txt --out resume.json
   python -m jobflow.cli crawl --config jobflow/config.yaml --cache ./.cache
   python -m jobflow.cli normalize --cache ./.cache --out jobs.csv
   python -m jobflow.cli match --resume resume.json --jobs jobs.csv --out matches.csv
   python -m jobflow.cli report --matches matches.csv --limit 10
   ```

   These commands follow the stages shown in the CLI doc: parsing a résumé, crawling job pages, normalizing them, matching, and generating a report.

So yes—you can store your API keys in a local `.env` file and load them before running the CLI. The tool will pick up `OPENAI_API_KEY` or `GEMINI_API_KEY` automatically, or fall back to a placeholder if none are set.
