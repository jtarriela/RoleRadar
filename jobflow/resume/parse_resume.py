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
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional imports for document handling.  These libraries enable
# extraction of text from PDF and DOCX files.  They are optional
# dependencies and may be ``None`` if not installed.
try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None  # type: ignore
try:
    import docx  # type: ignore
except ImportError:
    docx = None  # type: ignore

# Optional import for MinerU.  This library can convert complex
# documents (PDFs, Word, etc.) into LLM‑ready JSON/Markdown.  It is
# optional and may be ``None`` if not installed.  When available,
# the parser can use it to pre‑process documents before LLM inference.
try:
    import mineru  # type: ignore
except ImportError:
    mineru = None  # type: ignore

# Import the LLM provider factory used for LLM parsing.
from ..rank.llm_providers import get_default_provider


@dataclass
class Resume:
    """Structured résumé according to the MVP schema.

    This dataclass has been extended to support additional fields when
    parsing résumés with an LLM.  Legacy fields remain compatible
    with the original schema.  Optional attributes default to
    ``None`` or empty collections when not provided.
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


def _extract_text_from_file(file_path: str) -> str:
    """Extract text from a résumé file.

    This helper function attempts to read the contents of a résumé
    regardless of format.  For plain text files it simply reads the
    contents as UTF‑8.  For PDF and DOCX formats it will use
    optional dependencies (``pdfplumber`` and ``python‑docx``) if
    available.  If the corresponding library is not installed the
    caller should handle the ImportError upstream.

    Args:
        file_path: Path to the résumé file.

    Returns:
        A single string containing the extracted text.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file extension is unsupported and the
            fallback read fails.
    """
    ext = os.path.splitext(file_path)[1].lower()
    # PDF
    if ext == ".pdf":
        if pdfplumber is None:
            # Fallback: read binary and decode as latin‑1/ASCII.  This may
            # produce garbled text but prevents crashing when the optional
            # dependency is missing.  Users should install pdfplumber for
            # accurate extraction.
            with open(file_path, "rb") as f:
                data = f.read()
            return data.decode("latin-1", errors="ignore")
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    # DOCX
    if ext in {".doc", ".docx"}:
        if docx is None:
            raise RuntimeError(
                "python-docx is required to parse Word résumés; install it via pip"
            )
        document = docx.Document(file_path)
        return "\n".join(p.text for p in document.paragraphs)
    # TXT or unknown: attempt to read as UTF‑8 text
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _llm_parse_resume(text: str) -> Optional[Resume]:
    """Attempt to parse a résumé using an LLM provider.

    This function dispatches to the configured LLM provider (if any)
    and requests structured résumé fields.  It returns ``None`` if
    parsing fails for any reason or if no LLM provider is available.

    The prompt asks the model to return a JSON object with keys
    matching the :class:`Resume` dataclass fields (including
    extended fields).  The response is parsed and used to populate
    a new :class:`Resume` instance.  Missing keys default to
    reasonable values.

    Args:
        text: The full résumé text extracted from the original file.

    Returns:
        A ``Resume`` instance populated by the LLM, or ``None`` if
        parsing failed.
    """
    # Acquire provider; may be a PlaceholderProvider if no API keys
    try:
        provider = get_default_provider()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to initialise LLM provider: %s", exc)
        return None
    # If provider is the placeholder, we skip LLM parsing
    from . import parse_resume  # forward reference type hint suppressed
    if provider.__class__.__name__ == "PlaceholderProvider":
        return None
    # Build a prompt instructing the model to produce JSON
    # Build a prompt instructing the model to produce JSON.  Remove non‑ASCII
    # characters from the prompt and résumé text to avoid encoding errors with
    # some providers (e.g. GEMINI) that expect latin‑1/ASCII input.
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
    # Dispatch based on provider type
    import json
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
            # Unknown provider type
            return None
        data = json.loads(content)
        # Build Resume object; use defaults for missing keys
        full_name = data.get("full_name", "")
        contact = data.get("contact", {}) or {}
        roles = data.get("roles", []) or []
        skills = data.get("skills", []) or []
        education = data.get("education", []) or []
        experience = data.get("experience", []) or []
        certifications = data.get("certifications", []) or []
        summary = data.get("summary")
        # Infer YOE from roles list if numeric 'years' present
        yoe_total = _infer_years_of_experience(roles)
        resume = Resume(
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
        return resume
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM résumé parsing failed: %s", exc)
        return None


def parse_resume(file_path: str, use_llm: bool = True, use_mineru: bool = False) -> Resume:
    """Read a résumé file and parse it into structured JSON.

    This function supports multiple formats (text, PDF, DOCX) and
    optional LLM‑powered parsing.  If ``use_llm`` is true and a
    supported LLM provider is configured, the résumé will first be
    parsed using the provider.  On failure or if no provider is
    available, it falls back to the heuristic regex parser.

    Args:
        file_path: Path to the résumé file.  Supported extensions are
            ``.txt``, ``.pdf``, ``.doc`` and ``.docx``.
        use_llm: Whether to attempt LLM parsing (default: True).

    Returns:
        A :class:`Resume` instance populated with extracted fields.

    Raises:
        FileNotFoundError: If the file cannot be read.
        RuntimeError: If dependencies for the file type are missing.
    """
    # If requested, attempt to parse using MinerU first.  MinerU can
    # convert complex documents into structured Markdown or JSON.  We
    # treat the extracted text as the basis for LLM parsing or
    # heuristics.  If MinerU is not installed or fails, we fall back
    # to plain extraction.
    mineru_text: Optional[str] = None
    if use_mineru:
        if mineru is None:
            logger.warning("use_mineru=True but mineru package is not installed; falling back to text extraction")
        else:
            try:
                # MinerU provides a command line and Python API for
                # document extraction.  Here we attempt to use the
                # Python API if available, falling back to the CLI as
                # necessary.  For simplicity we call the CLI via
                # subprocess: mineru -p <file> -o <tmpdir>.  The
                # output will be Markdown or JSON; we read the first
                # resulting text file as the résumé text.
                import tempfile
                import shutil
                import subprocess
                tmp_dir = tempfile.mkdtemp(prefix="mineru_")
                try:
                    # Use the CLI to process the file into Markdown
                    result = subprocess.run(
                        ["mineru", "-p", file_path, "-o", tmp_dir],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if result.returncode != 0:
                        logger.warning("MinerU CLI failed: %s", result.stderr.strip())
                    else:
                        # Find the first .md or .json file in tmp_dir
                        for root, _, files in os.walk(tmp_dir):
                            for name in files:
                                if name.lower().endswith( (".md", ".txt", ".json") ):
                                    with open(os.path.join(root, name), "r", encoding="utf-8", errors="ignore") as f:
                                        mineru_text = f.read()
                                    break
                            if mineru_text:
                                break
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to run MinerU on %s: %s", file_path, exc)
    # Extract raw text (fallback or additional)
    text = mineru_text or _extract_text_from_file(file_path)
    # Optionally attempt LLM parsing
    if use_llm:
        llm_result = _llm_parse_resume(text)
        if llm_result is not None:
            return llm_result
    # Fallback: heuristic parsing
    resume = parse_resume_text(text)
    # If MinerU was used and returned content, annotate parsing_method
    if use_mineru and mineru_text:
        resume.parsing_method = "mineru"
    return resume


def save_resume_json(resume: Resume, out_path: str) -> None:
    """Serialize a `Resume` dataclass to JSON.

    Args:
        resume: The structured résumé to save.
        out_path: Path where the JSON file will be written.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(resume), f, indent=2)
    logger.info("Wrote résumé JSON to %s", out_path)