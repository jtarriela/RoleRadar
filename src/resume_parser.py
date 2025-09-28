"""
Resume parsing utilities.

This module provides functions to convert a résumé document into a structured
`Resume` dataclass.  In contrast to the earlier implementation, the
pre‑processing step using MinerU and the heuristic regular expression parser
have been simplified.  By default, the parser will attempt to use a
configured large language model (LLM) provider to extract structured fields
directly from the raw résumé text.  The Gemini provider (via the Google
Generative AI SDK) is recommended and defaults to the ``gemini‑1.5‑pro``
model.  If no LLM provider is configured or the API call fails, a minimal
fallback heuristic parser is used.

The `Resume` dataclass captures the core fields described in the project
README.  It may be extended with additional attributes as needed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

# Optional imports for PDF and Word document handling.  These libraries
# enable extraction of text from binary file formats.  They are optional
# dependencies and may be ``None`` if not installed.
try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None  # type: ignore
try:
    import docx  # type: ignore
except ImportError:
    docx = None  # type: ignore

# Import the LLM provider factory used for LLM parsing.
from llm_providers import get_default_provider

logger = logging.getLogger(__name__)


@dataclass
class Resume:
    """Structured résumé representation.

    This dataclass follows the schema defined in the project README.  All
    attributes map directly to keys expected in the JSON output.  Optional
    fields default to ``None`` or empty collections when not provided.  The
    ``embedding`` field may be populated by downstream code.
    """

    full_name: str
    contact: Dict[str, str]
    roles: List[Dict[str, object]]
    skills: List[str]
    education: List[Dict[str, object]]
    yoe_total: int
    preferences: Dict[str, object]
    embedding: Optional[List[float]] = None
    # Extended fields for LLM parsing
    summary: Optional[str] = None
    experience: List[Dict[str, object]] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    parsing_method: str = "regex"


def _infer_years_of_experience(roles: List[Dict[str, object]]) -> int:
    """Infer total years of experience from a list of roles.

    Each role may include a ``years`` field.  Non‑numeric values are ignored.

    Args:
        roles: List of role dictionaries.

    Returns:
        Integer sum of numeric ``years`` values.
    """
    total = 0
    for role in roles:
        years = role.get("years")
        if isinstance(years, (int, float)):
            total += int(years)
    return total


def parse_resume_text(text: str) -> Resume:
    """Parse résumé text using simple heuristics.

    This fallback parser uses regular expressions to extract a few common
    résumé fields.  It is intentionally conservative and should only be
    invoked when no LLM provider is configured.

    Args:
        text: The raw text of the résumé.

    Returns:
        A :class:`Resume` instance populated with extracted fields.
    """
    # Extract full name: assume first non‑empty line is the name.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    full_name = lines[0] if lines else ""
    # Extract contact email/phone.
    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)
    phone_match = re.search(r"\+?\d[\d\s\-]{7,}\d", text)
    contact = {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
    }
    # Extract roles by looking for common job titles.
    role_titles = []
    for m in re.finditer(r"\b([A-Z][a-z]+\s+(Engineer|Manager|Developer|Analyst))\b", text):
        role_titles.append(m.group(1))
    roles: List[Dict[str, object]] = [
        {"title": t, "years": 1, "skills": []} for t in role_titles
    ] or []
    # Extract skills: look for comma separated list after "Skills" heading.
    skills_section = re.search(r"Skills[:\s]+([A-Za-z0-9,\s]+)", text)
    skills: List[str] = []
    if skills_section:
        skills = [s.strip() for s in skills_section.group(1).split(",") if s.strip()]
    # Extract education: look for degree patterns (e.g. "BS in X, 2020").
    education: List[Dict[str, object]] = []
    for m in re.finditer(r"(BS|MS|BA|PhD) in ([A-Za-z &]+),?\s*(\d{4})", text):
        degree, field_str, year = m.groups()
        education.append({"degree": degree, "field": field_str.strip(), "year": int(year)})
    # Infer total years of experience.
    yoe_total = _infer_years_of_experience(roles)
    return Resume(
        full_name=full_name,
        contact=contact,
        roles=roles,
        skills=skills,
        education=education,
        yoe_total=yoe_total,
        preferences={},
    )


def _extract_text_from_file(file_path: str) -> str:
    """Extract text from a résumé file.

    Supports plain text, PDF and DOC/DOCX files.  If optional dependencies are
    missing, PDF and Word files may be partially extracted or raise errors.

    Args:
        file_path: Path to the résumé file.

    Returns:
        Text extracted from the file.

    Raises:
        FileNotFoundError: If the file cannot be found.
        RuntimeError: For unsupported file types when dependencies are missing.
    """
    ext = os.path.splitext(file_path)[1].lower()
    # PDF handling
    if ext == ".pdf":
        if pdfplumber is None:
            # Attempt to decode binary data as latin‑1/ASCII as a last resort.
            with open(file_path, "rb") as f:
                data = f.read()
            return data.decode("latin-1", errors="ignore")
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    # DOC/DOCX handling
    if ext in {".doc", ".docx"}:
        if docx is None:
            raise RuntimeError(
                "python-docx is required to parse Word résumés; install it via pip"
            )
        document = docx.Document(file_path)
        return "\n".join(p.text for p in document.paragraphs)
    # Plain text or unknown: read as UTF‑8
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _llm_parse_resume(text: str) -> Optional[Resume]:
    """Attempt to parse a résumé using a configured LLM provider.

    The function constructs a prompt instructing the model to return a
    JSON object containing the résumé fields.  The default provider is
    determined by the environment and may be OpenAI or Gemini.  If the
    provider is unavailable or parsing fails, ``None`` is returned.

    Args:
        text: The raw résumé text.

    Returns:
        A :class:`Resume` populated from the LLM output, or ``None`` if
        parsing failed.
    """
    try:
        provider = get_default_provider()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to initialise LLM provider: %s", exc)
        return None
    # If the provider is a placeholder, skip LLM parsing
    if provider.__class__.__name__ == "PlaceholderProvider":
        return None
    # Build a concise prompt.  Some providers are sensitive to non‑ASCII text,
    # so we remove non‑ASCII characters.
    def _to_ascii(s: str) -> str:
        return s.encode("ascii", "ignore").decode("ascii")

    prompt = (
        "You are a resume parser. Given the text of a candidate resume, "
        "extract structured information and return a JSON object with the "
        "following keys: full_name (string), contact (object with email and phone), "
        "roles (list of objects with title, years, skills), skills (list of strings), "
        "education (list of objects with degree, field, year), experience (list of "
        "objects with title, company, duration, description), certifications (list of strings), "
        "summary (string). Return only the JSON object.\n\n"
        f"Resume text:\n{_to_ascii(text)}"
    )
    try:
        if provider.__class__.__name__ == "OpenAIProvider":
            # type: ignore[attr-defined]
            response = provider.openai.ChatCompletion.create(
                model=provider.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message["content"]  # type: ignore[assignment]
        elif provider.__class__.__name__ == "GeminiProvider":
            # type: ignore[attr-defined]
            response = provider.model.generate_content(prompt)
            content = response.text  # type: ignore[assignment]
        else:
            return None
        data = json.loads(content)
        full_name = data.get("full_name", "")
        contact = data.get("contact", {}) or {}
        roles = data.get("roles", []) or []
        skills = data.get("skills", []) or []
        education = data.get("education", []) or []
        experience = data.get("experience", []) or []
        certifications = data.get("certifications", []) or []
        summary = data.get("summary")
        yoe_total = _infer_years_of_experience(roles)
        return Resume(
            full_name=full_name,
            contact=contact,
            roles=roles,
            skills=skills,
            education=education,
            yoe_total=yoe_total,
            preferences={},
            summary=summary,
            experience=experience,
            certifications=certifications,
            parsing_method="llm",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM résumé parsing failed: %s", exc)
        return None


def parse_resume(file_path: str, use_llm: bool = True) -> Resume:
    """Read a résumé file and return a structured :class:`Resume`.

    The parser attempts to use an LLM to extract structured information when
    ``use_llm`` is True.  If no LLM provider is configured or the LLM call
    fails, it falls back to a simple heuristic parser.

    Args:
        file_path: Path to the résumé file.  Supported extensions are .txt,
            .pdf, .doc and .docx.
        use_llm: Whether to attempt LLM parsing (default: True).

    Returns:
        A populated :class:`Resume` object.
    """
    text = _extract_text_from_file(file_path)
    if use_llm:
        llm_result = _llm_parse_resume(text)
        if llm_result is not None:
            return llm_result
    # Fallback heuristic parser
    return parse_resume_text(text)


def save_resume_json(resume: Resume, out_path: str) -> None:
    """Serialize a :class:`Resume` to a JSON file.

    Args:
        resume: The structured résumé to save.
        out_path: Destination path for the JSON file.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(resume), f, indent=2)
    logger.info("Wrote résumé JSON to %s", out_path)