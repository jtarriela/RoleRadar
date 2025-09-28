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


# def _llm_parse_resume(text: str) -> Optional[Resume]:
#     """Attempt to parse a résumé using a configured LLM provider.

#     The function constructs a prompt instructing the model to return a
#     JSON object containing the résumé fields.  The default provider is
#     determined by the environment and may be OpenAI or Gemini.  If the
#     provider is unavailable or parsing fails, ``None`` is returned.

#     Args:
#         text: The raw résumé text.

#     Returns:
#         A :class:`Resume` populated from the LLM output, or ``None`` if
#         parsing failed.
#     """
#     try:
#         provider = get_default_provider()
#     except Exception as exc:  # noqa: BLE001
#         logger.warning("Unable to initialise LLM provider: %s", exc)
#         return None
#     # If the provider is a placeholder, skip LLM parsing
#     if provider.__class__.__name__ == "PlaceholderProvider":
#         return None
#     # Build a concise prompt.  Some providers are sensitive to non‑ASCII text,
#     # so we remove non‑ASCII characters.
#     def _to_ascii(s: str) -> str:
#         return s.encode("ascii", "ignore").decode("ascii")

#     prompt = (
#         "You are a resume parser. Given the text of a candidate resume, "
#         "extract structured information and return a JSON object with the "
#         "following keys: full_name (string), contact (object with email and phone), "
#         "roles (list of objects with title, years, skills), skills (list of strings), "
#         "education (list of objects with degree, field, year), experience (list of "
#         "objects with title, company, duration, description), certifications (list of strings), "
#         "summary (string). Return only the JSON object.\n\n"
#         f"Resume text:\n{_to_ascii(text)}"
#     )
#     try:
#         if provider.__class__.__name__ == "OpenAIProvider":
#             # type: ignore[attr-defined]
#             response = provider.openai.ChatCompletion.create(
#                 model=provider.model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0,
#             )
#             content = response.choices[0].message["content"]  # type: ignore[assignment]
#         elif provider.__class__.__name__ == "GeminiProvider":
#             # type: ignore[attr-defined]
#             response = provider.model.generate_content(prompt)
#             content = response.text  # type: ignore[assignment]
#         else:
#             return None
#         data = json.loads(content)
#         full_name = data.get("full_name", "")
#         contact = data.get("contact", {}) or {}
#         roles = data.get("roles", []) or []
#         skills = data.get("skills", []) or []
#         education = data.get("education", []) or []
#         experience = data.get("experience", []) or []
#         certifications = data.get("certifications", []) or []
#         summary = data.get("summary")
#         yoe_total = _infer_years_of_experience(roles)
#         return Resume(
#             full_name=full_name,
#             contact=contact,
#             roles=roles,
#             skills=skills,
#             education=education,
#             yoe_total=yoe_total,
#             preferences={},
#             summary=summary,
#             experience=experience,
#             certifications=certifications,
#             parsing_method="llm",
#         )
#     except Exception as exc:  # noqa: BLE001
#         logger.warning("LLM résumé parsing failed: %s", exc)
#         return None


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
        print("Using LLM to parse resume...")
        llm_result = _llm_parse_resume(text)
        if llm_result is not None:
            print("LLM parsing succeeded.")
            return llm_result
    # Fallback heuristic parser
    print("Using fallback regex parser...") 
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
    
    
#######################
#DEBUG  
#######################

def _llm_parse_resume(text: str) -> Optional[Resume]:
    """Attempt to parse a résumé using a configured LLM provider with enhanced debugging."""
    try:
        provider = get_default_provider()
    except Exception as exc:
        logger.warning("Unable to initialise LLM provider: %s", exc)
        return None
    
    # If the provider is a placeholder, skip LLM parsing
    if provider.__class__.__name__ == "PlaceholderProvider":
        return None
    
    # Build a concise prompt with better structure
    def _to_ascii(s: str) -> str:
        return s.encode("ascii", "ignore").decode("ascii")

    prompt = (
        "You are a resume parser. Extract information from the resume text and return ONLY a valid JSON object.\n\n"
        "Return this exact JSON structure (no other text):\n"
        "{\n"
        '  "full_name": "string",\n'
        '  "contact": {"email": "string", "phone": "string"},\n'
        '  "roles": [{"title": "string", "years": number, "skills": ["string"]}],\n'
        '  "skills": ["string"],\n'
        '  "education": [{"degree": "string", "field": "string", "year": number}],\n'
        '  "experience": [{"title": "string", "company": "string", "duration": "string", "description": "string"}],\n'
        '  "certifications": ["string"],\n'
        '  "summary": "string"\n'
        "}\n\n"
        f"Resume text:\n{_to_ascii(text)[:10000000]}"  # Limit to 4000 chars to avoid token limits
    )
    
    try:
        print("🔍 DEBUG: Making LLM request...")
        print(f"Provider type: {provider.__class__.__name__}")
        
        if provider.__class__.__name__ == "OpenAIProvider":
            response = provider.client.chat.completions.create(
                model=provider.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            
        elif provider.__class__.__name__ == "GeminiProvider":
            response = provider.model.generate_content(prompt)
            content = response.text
            
        else:
            print(f"❌ Unknown provider type: {provider.__class__.__name__}")
            return None
        
        # 🚨 DEBUG: Print the raw response
        print("\n" + "="*60)
        print("🔍 RAW LLM RESPONSE:")
        print("="*60)
        print(f"Response type: {type(content)}")
        print(f"Response length: {len(content) if content else 0}")
        print("Raw content:")
        print(repr(content))  # This shows exact characters including whitespace
        print("\nFormatted content:")
        print(content)
        print("="*60)
        
        # Check if response is empty or None
        if not content:
            print("❌ Empty response from LLM")
            return None
        
        # Enhanced cleaning with debug output
        print("\n🧹 CLEANING RESPONSE...")
        original_content = content
        
        # Remove common markdown formatting
        content = content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "", 1)
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()
        
        # Find JSON boundaries
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx >= 0 and end_idx >= 0:
            content = content[start_idx:end_idx + 1]
        
        print(f"Original length: {len(original_content)}")
        print(f"Cleaned length: {len(content)}")
        print("Cleaned content:")
        print(repr(content))
        
        # Try to parse JSON with detailed error info
        try:
            print("\n🔧 PARSING JSON...")
            data = json.loads(content)
            print("✅ JSON parsing successful!")
            print(f"Keys found: {list(data.keys())}")
            
        except json.JSONDecodeError as json_err:
            print(f"❌ JSON parsing failed: {json_err}")
            print(f"Error at position: {json_err.pos}")
            print(f"Content around error: {content[max(0, json_err.pos-20):json_err.pos+20]}")
            
            # Try to fix common JSON issues
            print("\n🔧 Attempting JSON repair...")
            
            # Fix common issues
            fixed_content = content
            
            # Replace single quotes with double quotes
            fixed_content = fixed_content.replace("'", '"')
            
            # Fix trailing commas
            import re
            fixed_content = re.sub(r',(\s*[}\]])', r'\1', fixed_content)
            
            # Try parsing the fixed version
            try:
                data = json.loads(fixed_content)
                print("✅ JSON repair successful!")
                content = fixed_content
            except json.JSONDecodeError as repair_err:
                print(f"❌ JSON repair failed: {repair_err}")
                return None
        
        # Create Resume object
        full_name = data.get("full_name", "")
        contact = data.get("contact", {}) or {}
        roles = data.get("roles", []) or []
        skills = data.get("skills", []) or []
        education = data.get("education", []) or []
        experience = data.get("experience", []) or []
        certifications = data.get("certifications", []) or []
        summary = data.get("summary")
        yoe_total = _infer_years_of_experience(roles)
        
        print(f"\n✅ Successfully created Resume object:")
        print(f"   Name: {full_name}")
        print(f"   Email: {contact.get('email', 'Not found')}")
        print(f"   Skills: {len(skills)} found")
        print(f"   Experience: {len(experience)} entries")
        
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
        
    except Exception as exc:
        print(f"❌ LLM résumé parsing failed: {exc}")
        logger.warning("LLM résumé parsing failed: %s", exc)
        import traceback
        traceback.print_exc()
        return None


# Quick test function
def debug_gemini_directly():
    """Test Gemini API directly to see what's happening."""
    import os
    import google.generativeai as genai
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return
    
    genai.configure(api_key=api_key)
    import llm_providers
    model = llm_providers.get_default_provider()
    
    # Simple test prompt
    test_prompt = """
    Return only this JSON object (no other text):
    {"name": "Test User", "skills": ["Python", "JavaScript"]}
    """
    
    # print("🧪 Testing Gemini directly...")
    # try:
    #     response = model.generate_content(test_prompt)
    #     print(f"Response type: {type(response.text)}")
    #     print(f"Response: {repr(response.text)}")
        
    #     # Try parsing
    #     import json
    #     data = json.loads(response.text.strip())
    #     print("✅ Direct Gemini test successful!")
        
    # except Exception as e:
    #     print(f"❌ Direct Gemini test failed: {e}")
    #     import traceback
    #     traceback.print_exc()
        
if __name__ == "__main__":
    
    path = "/Users/jdtarriela/Documents/git/RoleRadar/jtarriela_resume[sp].pdf"
    
    resume = parse_resume(path, use_llm=True)
    print(resume)
    save_resume_json(resume, "jtarriela_resume.json")
    
    # debug_gemini_directly()

    '''
    resume path
    resume parser instantiate
    resume parse resume 
    save json to file 
    '''