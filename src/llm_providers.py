"""
Enhanced LLM provider abstractions with job parsing capabilities.

This module extends the original LLM providers to include job markdown parsing
functionality, converting scraped job postings into structured JSON format.
"""

from __future__ import annotations

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def judge(self, resume_json: Dict[str, object], job_text: str, job_title: str, company: str) -> Tuple[float, List[str], List[str]]:
        """Judge the relevance of a résumé to a job posting."""
        raise NotImplementedError

    @abstractmethod
    def parse_job_markdown(self, markdown_content: str, job_url: str = "") -> Dict[str, object]:
        """Parse job markdown into structured JSON format."""
        raise NotImplementedError

    def _create_job_parsing_prompt(self, markdown_content: str, job_url: str = "") -> str:
        """Create a consistent prompt for job parsing across providers."""
        return f"""
You are a job posting parser. Extract structured information from the following job posting markdown and return a JSON object with the exact structure shown below.

IMPORTANT: Return ONLY valid JSON, no additional text or explanation.

Required JSON structure:
{{
  "job_info": {{
    "title": "extracted job title",
    "company": "company name", 
    "location": "location (city, state/country)",
    "employment_type": "Full-time/Part-time/Contract/Internship",
    "salary_range": "salary range if mentioned, empty string if not",
    "job_url": "{job_url}",
    "posted_date": "date if found, empty string if not"
  }},
  "requirements": {{
    "experience_level": "Entry/Mid/Senior/Executive level",
    "education": ["degree requirements"],
    "required_skills": ["list of must-have skills"],
    "preferred_skills": ["list of nice-to-have skills"], 
    "certifications": ["any certifications mentioned"]
  }},
  "description": {{
    "summary": "brief job summary",
    "responsibilities": ["list of key responsibilities"],
    "benefits": ["benefits and perks mentioned"],
    "keywords": ["important keywords from posting"]
  }},
  "parsing_metadata": {{
    "confidence_score": 0.95,
    "timestamp": "{datetime.now().isoformat()}",
    "source_quality": "high/medium/low"
  }}
}}

Job posting markdown:
{markdown_content}
"""


class PlaceholderProvider(LLMProvider):
    """Fallback provider that does not call any external API."""

    def judge(self, resume_json: Dict[str, object], job_text: str, job_title: str, company: str) -> Tuple[float, List[str], List[str]]:
        skills = set(resume_json.get("skills", []))
        missing = [s for s in skills if s.lower() not in job_text.lower()]
        reasons = [
            f"Basic role alignment with {job_title}",
            f"Company {company} and candidate skills overlap",
        ]
        relevance = 0.5
        return relevance, reasons, missing

    def parse_job_markdown(self, markdown_content: str, job_url: str = "") -> Dict[str, object]:
        """Basic text parsing fallback when no LLM is available."""
        lines = markdown_content.split('\n')
        
        # Try to extract basic info with simple heuristics
        title = "Unknown Position"
        company = "Unknown Company" 
        location = "Unknown Location"
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 5 and len(line) < 100:  # Likely title
                if any(word in line.lower() for word in ['engineer', 'developer', 'manager', 'analyst', 'specialist']):
                    title = line
                    break
        
        return {
            "job_info": {
                "title": title,
                "company": company,
                "location": location,
                "employment_type": "Full-time",
                "salary_range": "",
                "job_url": job_url,
                "posted_date": ""
            },
            "requirements": {
                "experience_level": "Mid-level",
                "education": [],
                "required_skills": [],
                "preferred_skills": [],
                "certifications": []
            },
            "description": {
                "summary": "Job details extracted from markdown (limited parsing)",
                "responsibilities": [],
                "benefits": [],
                "keywords": []
            },
            "parsing_metadata": {
                "confidence_score": 0.3,
                "timestamp": datetime.now().isoformat(),
                "source_quality": "low"
            }
        }


class OpenAIProvider(LLMProvider):
    """Provider that uses the OpenAI ChatCompletion API."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-3.5-turbo") -> None:
        try:
            import openai
            self.openai = openai
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OpenAIProvider. Install it via pip."
            ) from exc
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        
        # Configure client (adjust based on openai version)
        try:
            # For newer openai versions (1.x+)
            self.client = openai.OpenAI(api_key=self.api_key)
        except AttributeError:
            # For older versions
            openai.api_key = self.api_key
            self.client = None

    def judge(self, resume_json: Dict[str, object], job_text: str, job_title: str, company: str) -> Tuple[float, List[str], List[str]]:
        skills = ", ".join(resume_json.get("skills", []))
        prompt = (
            "You are a recruiting assistant. Given a candidate résumé and a job description, "
            "assess how well the candidate fits the role. Return a JSON object with keys "
            "'score' (0 to 1), 'reasons' (list of strings explaining the match), and 'missing' "
            "(list of required skills missing from the job description).\n"
            f"Candidate skills: {skills}\n"
            f"Job Title: {job_title}\n"
            f"Company: {company}\n"
            f"Job Description: {job_text}"
        )
        
        try:
            if self.client:  # New API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
            else:  # Old API
                response = self.openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message["content"]
                
            data = json.loads(content)
            score = float(data.get("score", 0.0))
            reasons = data.get("reasons", [])
            missing = data.get("missing", [])
            return score, reasons, missing
            
        except Exception as exc:
            logger.exception("OpenAI API call failed: %s", exc)
            placeholder = PlaceholderProvider()
            return placeholder.judge(resume_json, job_text, job_title, company)

    def parse_job_markdown(self, markdown_content: str, job_url: str = "") -> Dict[str, object]:
        """Parse job markdown using OpenAI."""
        prompt = self._create_job_parsing_prompt(markdown_content, job_url)
        
        try:
            if self.client:  # New API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
            else:  # Old API
                response = self.openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message["content"]
            
            # Parse JSON response
            parsed_job = json.loads(content)
            
            # Ensure job_url is set
            parsed_job["job_info"]["job_url"] = job_url
            
            logger.info(f"Successfully parsed job: {parsed_job['job_info']['title']}")
            return parsed_job
            
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse OpenAI JSON response: {exc}")
            placeholder = PlaceholderProvider()
            return placeholder.parse_job_markdown(markdown_content, job_url)
        except Exception as exc:
            logger.exception(f"OpenAI job parsing failed: {exc}")
            placeholder = PlaceholderProvider()
            return placeholder.parse_job_markdown(markdown_content, job_url)


class GeminiProvider(LLMProvider):
    """Provider that uses Google Generative AI (Gemini)."""

    def __init__(self, api_key: str | None = None, model: str = os.getenv("GEMINI_MODEL")) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai package is required for GeminiProvider. Install it via pip."
            ) from exc
        
        self.genai = genai
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        env_model = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL")
        self.model_name = env_model or model
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY not provided")
        
        self.genai.configure(api_key=self.api_key)
        try:
            self.model = self.genai.GenerativeModel(self.model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Gemini model {self.model_name}: {exc}")

    def judge(self, resume_json: Dict[str, object], job_text: str, job_title: str, company: str) -> Tuple[float, List[str], List[str]]:
        skills = ", ".join(resume_json.get("skills", []))
        prompt = (
            "You are a recruiting assistant. Given a candidate résumé and a job description, "
            "assess how well the candidate fits the role. Return a JSON object with keys "
            "'score' (0 to 1), 'reasons' (list of strings explaining the match), and 'missing' "
            "(list of required skills missing from the job description).\n"
            f"Candidate skills: {skills}\n"
            f"Job Title: {job_title}\n"
            f"Company: {company}\n"
            f"Job Description: {job_text}"
        )
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            data = json.loads(content)
            score = float(data.get("score", 0.0))
            reasons = data.get("reasons", [])
            missing = data.get("missing", [])
            return score, reasons, missing
        except Exception as exc:
            logger.exception("Gemini API call failed: %s", exc)
            placeholder = PlaceholderProvider()
            return placeholder.judge(resume_json, job_text, job_title, company)

    def parse_job_markdown(self, markdown_content: str, job_url: str = "") -> Dict[str, object]:
        """Parse job markdown using Gemini."""
        prompt = self._create_job_parsing_prompt(markdown_content, job_url)
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Clean response if it has markdown formatting
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            parsed_job = json.loads(content)
            parsed_job["job_info"]["job_url"] = job_url
            
            logger.info(f"Successfully parsed job: {parsed_job['job_info']['title']}")
            return parsed_job
            
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse Gemini JSON response: {exc}")
            placeholder = PlaceholderProvider()
            return placeholder.parse_job_markdown(markdown_content, job_url)
        except Exception as exc:
            logger.exception(f"Gemini job parsing failed: {exc}")
            placeholder = PlaceholderProvider()
            return placeholder.parse_job_markdown(markdown_content, job_url)


def get_default_provider() -> LLMProvider:
    """Return an LLMProvider instance based on configuration and API keys."""
    preferred = os.getenv("LLM_PROVIDER")
    if preferred:
        pref = preferred.lower()
        if pref == "openai":
            try:
                return OpenAIProvider(os.getenv("OPENAI_API_KEY"))
            except Exception as exc:
                logger.warning("LLM_PROVIDER=openai but failed to initialise OpenAIProvider: %s", exc)
        elif pref == "gemini":
            try:
                return GeminiProvider(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            except Exception as exc:
                logger.warning("LLM_PROVIDER=gemini but failed to initialise GeminiProvider: %s", exc)
        elif pref == "placeholder":
            logger.info("LLM_PROVIDER=placeholder; using placeholder provider")
            return PlaceholderProvider()
        else:
            logger.warning("Unknown LLM_PROVIDER value '%s'; falling back to automatic detection", preferred)
    
    api_key_openai = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if api_key_openai:
        try:
            return OpenAIProvider(api_key_openai)
        except Exception as exc:
            logger.warning("Failed to initialise OpenAIProvider: %s", exc)
    
    if api_key_gemini:
        try:
            return GeminiProvider(api_key_gemini)
        except Exception as exc:
            logger.warning("Failed to initialise GeminiProvider: %s", exc)
    
    logger.info("No LLM API keys found; using placeholder provider")
    return PlaceholderProvider()