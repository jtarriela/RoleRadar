"""
Resume parser.

This module provides simple functions to parse a résumé into a
structured JSON format.  The MVP specification defines a strict
schema for the resulting JSON (see README).  A real implementation
would delegate to an LLM to extract fields like contact info,
roles, skills and education.  Here we provide a minimal heuristic
parser for demonstration and leave hooks for plugging in an LLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Resume:
    """Structured résumé according to the MVP schema."""
    full_name: str
    contact: Dict[str, str]
    roles: List[Dict[str, object]]
    skills: List[str]
    education: List[Dict[str, object]]
    yoe_total: int
    preferences: Dict[str, object]
    embedding: Optional[List[float]] = None


def _infer_years_of_experience(roles: List[Dict[str, object]]) -> int:
    """Infer total years of experience from role years.

    Args:
        roles: List of roles each containing a `years` field.

    Returns:
        Integer sum of role years, capped at a reasonable maximum.
    """
    total = 0
    for role in roles:
        years = role.get("years")
        if isinstance(years, (int, float)):
            total += int(years)
    return total


def parse_resume_text(text: str) -> Resume:
    """Parse résumé text into a structured `Resume` dataclass.

    This heuristic implementation searches for typical résumé
    sections.  It does not handle all formats and should be
    replaced with an LLM call for production use.
    """
    # Extract full name: assume first line is the name.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    full_name = lines[0] if lines else ""
    # Extract contact email/phone.
    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)
    phone_match = re.search(r"\+?\d[\d\s\-]{7,}\d", text)
    contact = {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
    }
    # Dummy roles: look for keywords like "Engineer", "Manager".
    role_titles = []
    for m in re.finditer(r"\b([A-Z][a-z]+\s+(Engineer|Manager|Developer|Analyst))\b", text):
        title = m.group(1)
        role_titles.append(title)
    roles: List[Dict[str, object]] = [
        {"title": t, "years": 1, "skills": []} for t in role_titles
    ] or []
    # Extract skills: comma separated keywords after "Skills" heading.
    skills_section = re.search(r"Skills[:\s]+([A-Za-z0-9,\s]+)", text)
    skills: List[str] = []
    if skills_section:
        skills = [s.strip() for s in skills_section.group(1).split(",") if s.strip()]
    # Extract education: look for lines containing degree patterns (e.g. "BS", "MS").
    education: List[Dict[str, object]] = []
    for m in re.finditer(r"(BS|MS|BA|PhD) in ([A-Za-z &]+),?\s*(\d{4})", text):
        degree, field, year = m.groups()
        education.append({"degree": degree, "field": field.strip(), "year": int(year)})
    # Infer total years of experience.
    yoe_total = _infer_years_of_experience(roles)
    # Default preferences: none specified.
    preferences = {}
    resume = Resume(
        full_name=full_name,
        contact=contact,
        roles=roles,
        skills=skills,
        education=education,
        yoe_total=yoe_total,
        preferences=preferences,
    )
    logger.debug("Parsed résumé: %s", resume)
    return resume


def parse_resume(file_path: str) -> Resume:
    """Read a résumé file and parse it into structured JSON.

    Currently only plain text and `.txt` files are supported.  For
    PDFs or Word documents consider using the MegaParse or MinerU
    libraries (see README) to extract text before calling this
    function.
    """
    _, ext = os.path.splitext(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_resume_text(text)


def save_resume_json(resume: Resume, out_path: str) -> None:
    """Serialize a `Resume` dataclass to JSON.

    Args:
        resume: The structured résumé to save.
        out_path: Path where the JSON file will be written.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(resume), f, indent=2)
    logger.info("Wrote résumé JSON to %s", out_path)