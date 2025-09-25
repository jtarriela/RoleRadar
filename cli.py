# cli.py
import csv
from pathlib import Path
import typer
from typing import Optional, List
from normalize.schema import JOB_HEADERS, JobRow
from ingest.greenhouse_adapter import fetch as gh_fetch
from ingest.lever_adapter import fetch as lv_fetch

app = typer.Typer(add_completion=False, help="jobflow — crawl & normalize jobs to CSV")

def write_rows(rows: List[JobRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(JOB_HEADERS)
        for r in rows:
            w.writerow(r.to_csv_row())

@app.command("gh")
def greenhouse(
    token: str = typer.Argument(..., help="Greenhouse board token, e.g. 'openai'"),
    out: Path = typer.Option(Path("./outputs/jobs.csv"), "--out", "-o", help="CSV output file")
):
    """Fetch jobs from Greenhouse and append to CSV."""
    rows = gh_fetch(token)
    write_rows(rows, out)
    typer.echo(f"[OK] Wrote {len(rows)} Greenhouse rows → {out}")

@app.command("lever")
def lever(
    slug: str = typer.Argument(..., help="Lever company slug, e.g. 'databricks'"),
    out: Path = typer.Option(Path("./outputs/jobs.csv"), "--out", "-o", help="CSV output file")
):
    """Fetch jobs from Lever and append to CSV."""
    rows = lv_fetch(slug)
    write_rows(rows, out)
    typer.echo(f"[OK] Wrote {len(rows)} Lever rows → {out}")

if __name__ == "__main__":
    app()
