#!/usr/bin/env python3
"""
Hybrid Job-Resume Matcher: Structured Field Extraction + Semantic Similarity
This actually checks years of experience, specific skills, education requirements, etc.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from dataclasses import dataclass, asdict
import logging
import os
import re
from datetime import datetime, date
from tqdm import tqdm
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StructuredMatchBreakdown:
    """Detailed breakdown of match components"""
    experience_match: float
    experience_details: str
    skill_match: float
    skill_details: Dict
    education_match: float
    education_details: str
    semantic_match: float
    overall_score: float
    recommendation: str
    missing_requirements: List[str]
    improvement_suggestions: List[str]

@dataclass
class JobMatchResult:
    """Complete job match result with breakdown"""
    job_title: str
    company: str
    match_score: float
    job_url: str
    salary_range: str
    location: str
    recommendation: str
    match_breakdown: StructuredMatchBreakdown

class StructuredJobMatcher:
    """
    Hybrid matcher that combines structured field analysis with semantic similarity
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 100):
        """
        Initialize with sentence transformer and configuration
        
        Args:
            model_name: Sentence transformer model for semantic similarity
            batch_size: Batch size for processing multiple jobs
        """
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        
        # Cache for resume data to avoid recomputation
        self.resume_cache = {}
        
        # Scoring weights (tune these based on importance)
        self.weights = {
            'experience': 0.25,    # 25% - Years of experience
            'skills': 0.45,        # 45% - Required skills match (most important)
            'education': 0.10,     # 10% - Education requirements
            'semantic': 0.20       # 20% - Overall contextual fit
        }
        
        logger.info(f"Initialized with weights: {self.weights}")
    
    def extract_resume_structured_data(self, resume_json: Dict) -> Dict:
        """
        Extract structured, measurable data from resume
        
        Returns:
            Dict with years_experience, skills, education_level, domains, etc.
        """
        resume_id = resume_json.get('full_name', 'default')
        
        if resume_id in self.resume_cache:
            return self.resume_cache[resume_id]
        
        # 1. Calculate total years of experience
        total_years = self.calculate_total_experience(resume_json)
        
        # 2. Extract all skills (flatten all skill categories)
        all_skills = self.extract_all_skills(resume_json)
        
        # 3. Determine education level
        education_level = self.extract_education_level(resume_json)
        
        # 4. Extract domain experience
        domains = self.extract_domain_experience(resume_json)
        
        # 5. Extract key technologies/tools
        technologies = self.extract_technologies(resume_json)
        
        structured_data = {
            'years_experience': total_years,
            'skills': all_skills,
            'education_level': education_level,
            'domains': domains,
            'technologies': technologies,
            'certifications': resume_json.get('certifications', []),
            'most_recent_role': self.get_most_recent_role(resume_json)
        }
        
        # Cache for reuse
        self.resume_cache[resume_id] = structured_data
        
        logger.info(f"Extracted resume data: {total_years} years exp, {len(all_skills)} skills, {education_level} education")
        
        return structured_data
    
    def extract_job_requirements(self, job_json: Dict) -> Dict:
        """
        Extract structured requirements from job posting
        
        Returns:
            Dict with min_years, required_skills, education_req, etc.
        """
        requirements = job_json.get('requirements', {})
        
        # 1. Parse minimum years required
        min_years = self.parse_experience_requirement(requirements.get('experience_level', ''))
        
        # 2. Get required and preferred skills
        required_skills = requirements.get('required_skills', [])
        preferred_skills = requirements.get('preferred_skills', [])
        
        # 3. Parse education requirements
        education_req = self.parse_education_requirements(requirements.get('education', []))
        
        # 4. Extract key technologies mentioned in job
        job_technologies = self.extract_job_technologies(job_json)
        
        # 5. Determine seniority level
        seniority = self.determine_seniority_level(requirements.get('experience_level', ''))
        
        return {
            'min_years_required': min_years,
            'required_skills': required_skills,
            'preferred_skills': preferred_skills,
            'education_required': education_req,
            'technologies_mentioned': job_technologies,
            'seniority_level': seniority,
            'experience_level_text': requirements.get('experience_level', '')
        }
    
    def calculate_experience_match(self, resume_data: Dict, job_requirements: Dict) -> Tuple[float, str]:
        """
        Calculate how well experience level matches
        
        Returns:
            (match_percentage, explanation_string)
        """
        resume_years = resume_data['years_experience']
        required_years = job_requirements['min_years_required']
        
        if required_years == 0:
            # Entry level position
            match_score = 100.0
            explanation = f"Entry level position - {resume_years} years qualifies"
        elif resume_years >= required_years:
            # Meets or exceeds requirement
            if resume_years <= required_years * 1.5:
                match_score = 100.0
                explanation = f"Perfect fit: {resume_years} years vs {required_years} required"
            else:
                # Might be overqualified
                match_score = 90.0
                explanation = f"Overqualified: {resume_years} years vs {required_years} required"
        else:
            # Under-qualified
            match_score = (resume_years / required_years) * 80  # Max 80% if under-qualified
            explanation = f"Under-qualified: {resume_years} years vs {required_years} required"
        
        return match_score, explanation
    
    def calculate_skills_match(self, resume_data: Dict, job_requirements: Dict) -> Tuple[float, Dict]:
        """
        Calculate detailed skills matching
        
        Returns:
            (match_percentage, detailed_breakdown_dict)
        """
        resume_skills = set(skill.lower().strip() for skill in resume_data['skills'])
        resume_technologies = set(tech.lower().strip() for tech in resume_data['technologies'])
        all_resume_capabilities = resume_skills.union(resume_technologies)
        
        required_skills = job_requirements['required_skills']
        preferred_skills = job_requirements['preferred_skills']
        
        # Match required skills
        matched_required = []
        missing_required = []
        
        for req_skill in required_skills:
            if self.skill_matches_resume(req_skill, all_resume_capabilities):
                matched_required.append(req_skill)
            else:
                missing_required.append(req_skill)
        
        # Match preferred skills
        matched_preferred = []
        missing_preferred = []
        
        for pref_skill in preferred_skills:
            if self.skill_matches_resume(pref_skill, all_resume_capabilities):
                matched_preferred.append(pref_skill)
            else:
                missing_preferred.append(pref_skill)
        
        # Calculate score
        if len(required_skills) == 0:
            required_match_pct = 100.0
        else:
            required_match_pct = (len(matched_required) / len(required_skills)) * 100
        
        if len(preferred_skills) == 0:
            preferred_match_pct = 0.0
        else:
            preferred_match_pct = (len(matched_preferred) / len(preferred_skills)) * 100
        
        # Weighted skills score: required skills are more important
        overall_skills_score = (required_match_pct * 0.8) + (preferred_match_pct * 0.2)
        
        skills_breakdown = {
            'required_skills_match': required_match_pct,
            'preferred_skills_match': preferred_match_pct,
            'matched_required': matched_required,
            'missing_required': missing_required,
            'matched_preferred': matched_preferred,
            'missing_preferred': missing_preferred,
            'total_required': len(required_skills),
            'total_preferred': len(preferred_skills)
        }
        
        return overall_skills_score, skills_breakdown
    
    def skill_matches_resume(self, required_skill: str, resume_capabilities: set) -> bool:
        """
        Check if a required skill matches anything in resume
        Uses fuzzy matching to handle variations
        """
        required_lower = required_skill.lower().strip()
        
        # Direct match
        if required_lower in resume_capabilities:
            return True
        
        # Check if any words from the required skill appear in resume
        required_words = re.findall(r'\b\w+\b', required_lower)
        
        # Remove common words that don't indicate skill match
        stop_words = {'experience', 'with', 'using', 'knowledge', 'of', 'in', 'and', 'or', 'the', 'a', 'an'}
        meaningful_words = [word for word in required_words if word not in stop_words and len(word) > 2]
        
        # Check if any meaningful words match
        for word in meaningful_words:
            if any(word in capability for capability in resume_capabilities):
                return True
        
        return False
    
    def calculate_education_match(self, resume_data: Dict, job_requirements: Dict) -> Tuple[float, str]:
        """
        Calculate education requirements match
        
        Returns:
            (match_percentage, explanation_string)
        """
        resume_education = resume_data['education_level']
        required_education = job_requirements['education_required']
        
        # Education hierarchy
        education_levels = {
            'None': 0,
            'High School': 1,
            'Associates': 2,
            'Bachelors': 3,
            'Masters': 4,
            'PhD': 5
        }
        
        resume_level = education_levels.get(resume_education, 0)
        required_level = education_levels.get(required_education, 0)
        
        if required_level == 0 or resume_level >= required_level:
            match_score = 100.0
            explanation = f"Education requirement met: {resume_education} vs {required_education} required"
        else:
            # Partial credit for being close
            match_score = max(0, (resume_level / required_level) * 100)
            explanation = f"Education gap: {resume_education} vs {required_education} required"
        
        return match_score, explanation
    
    def calculate_semantic_match(self, resume_json: Dict, job_json: Dict) -> float:
        """
        Calculate semantic similarity using sentence transformers
        This is your original approach, but now just one component
        """
        resume_text = self.format_resume_for_embedding(resume_json)
        job_text = self.format_job_for_embedding(job_json)
        
        # Generate embeddings
        resume_embedding = self.model.encode([resume_text])[0]
        job_embedding = self.model.encode([job_text])[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        
        return max(0, min(100, similarity * 100))
    
    def format_resume_for_embedding(self, resume_json: Dict) -> str:
        """Format resume for semantic analysis (optimized)"""
        sections = []
        
        if resume_json.get('summary'):
            sections.append(f"Professional Summary: {resume_json['summary']}")
        
        # Recent experience only for semantic analysis
        if resume_json.get('experience'):
            recent_exp = resume_json['experience'][:2]  # Most recent 2 roles
            for exp in recent_exp:
                desc = exp['description'][:300] + "..." if len(exp['description']) > 300 else exp['description']
                sections.append(f"{exp['title']}: {desc}")
        
        # Skills summary
        if resume_json.get('skills'):
            if isinstance(resume_json['skills'], list):
                skills_text = ', '.join(resume_json['skills'][:15])  # Top 15 skills
            else:
                all_skills = []
                for skill_category in resume_json['skills'].values():
                    if isinstance(skill_category, list):
                        all_skills.extend(skill_category)
                skills_text = ', '.join(all_skills[:15])
            sections.append(f"Skills: {skills_text}")
        
        return "\n\n".join(sections)
    
    def format_job_for_embedding(self, job_json: Dict) -> str:
        """Format job for semantic analysis (optimized)"""
        sections = []
        
        # Job title and summary
        if job_json.get('job_info'):
            sections.append(f"Position: {job_json['job_info'].get('title', '')}")
        
        if job_json.get('description', {}).get('summary'):
            summary = job_json['description']['summary'][:400]
            sections.append(f"Role: {summary}")
        
        # Key requirements
        if job_json.get('requirements'):
            req = job_json['requirements']
            if req.get('required_skills'):
                skills = ', '.join(req['required_skills'][:10])
                sections.append(f"Required: {skills}")
        
        # Top responsibilities
        if job_json.get('description', {}).get('responsibilities'):
            resp_list = job_json['description']['responsibilities'][:3]
            resp_text = ' '.join(resp_list)[:300]
            sections.append(f"Responsibilities: {resp_text}")
        
        return "\n\n".join(sections)
    
    def create_comprehensive_match(self, resume_json: Dict, job_json: Dict) -> JobMatchResult:
        """
        Create comprehensive match analysis combining all components
        """
        # Extract structured data
        resume_data = self.extract_resume_structured_data(resume_json)
        job_requirements = self.extract_job_requirements(job_json)
        
        # Calculate individual match components
        experience_score, experience_details = self.calculate_experience_match(resume_data, job_requirements)
        skills_score, skills_breakdown = self.calculate_skills_match(resume_data, job_requirements)
        education_score, education_details = self.calculate_education_match(resume_data, job_requirements)
        semantic_score = self.calculate_semantic_match(resume_json, job_json)
        
        # Calculate weighted overall score
        overall_score = (
            experience_score * self.weights['experience'] +
            skills_score * self.weights['skills'] +
            education_score * self.weights['education'] +
            semantic_score * self.weights['semantic']
        )
        
        # Generate recommendation
        recommendation = self.generate_recommendation(overall_score, skills_breakdown)
        
        # Create improvement suggestions
        improvement_suggestions = self.generate_improvement_suggestions(
            skills_breakdown, experience_score, education_score
        )
        
        # Create detailed breakdown
        match_breakdown = StructuredMatchBreakdown(
            experience_match=experience_score,
            experience_details=experience_details,
            skill_match=skills_score,
            skill_details=skills_breakdown,
            education_match=education_score,
            education_details=education_details,
            semantic_match=semantic_score,
            overall_score=overall_score,
            recommendation=recommendation,
            missing_requirements=skills_breakdown['missing_required'],
            improvement_suggestions=improvement_suggestions
        )
        
        # Extract job info
        job_info = job_json.get('job_info', {})
        
        return JobMatchResult(
            job_title=job_info.get('title', 'Unknown'),
            company=job_info.get('company', 'Unknown'),
            match_score=round(overall_score, 2),
            job_url=job_info.get('job_url', ''),
            salary_range=job_info.get('salary_range', 'Not specified'),
            location=job_info.get('location', 'Not specified'),
            recommendation=recommendation,
            match_breakdown=match_breakdown
        )
    
    def generate_recommendation(self, overall_score: float, skills_breakdown: Dict) -> str:
        """Generate recommendation based on score and missing requirements"""
        missing_required_count = len(skills_breakdown['missing_required'])
        
        if overall_score >= 80:
            return "🎯 EXCELLENT MATCH - Apply immediately!"
        elif overall_score >= 70:
            return "🚀 STRONG MATCH - Highly recommended"
        elif overall_score >= 60:
            if missing_required_count == 0:
                return "✅ GOOD MATCH - Consider applying"
            else:
                return "📝 GOOD POTENTIAL - Address missing skills first"
        elif overall_score >= 50:
            return "🤔 POTENTIAL MATCH - Significant improvements needed"
        else:
            return "❌ LOW MATCH - Focus on better aligned opportunities"
    
    def generate_improvement_suggestions(self, skills_breakdown: Dict, experience_score: float, education_score: float) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Skill-based suggestions
        missing_required = skills_breakdown['missing_required']
        if missing_required:
            top_missing = missing_required[:3]
            suggestions.append(f"Add these critical skills: {', '.join(top_missing)}")
        
        missing_preferred = skills_breakdown['missing_preferred']
        if missing_preferred and len(missing_preferred) <= 3:
            suggestions.append(f"Consider adding preferred skills: {', '.join(missing_preferred)}")
        
        # Experience suggestions
        if experience_score < 80:
            suggestions.append("Highlight relevant experience more prominently")
        
        # Education suggestions
        if education_score < 80:
            suggestions.append("Consider relevant certifications or additional education")
        
        return suggestions
    
    def process_job_batch_structured(self, resume_json: Dict, jobs_list: List[Dict]) -> List[JobMatchResult]:
        """
        Process jobs with structured matching approach
        """
        results = []
        
        logger.info(f"Processing {len(jobs_list):,} jobs with structured matching...")
        
        with tqdm(total=len(jobs_list), desc="Structured Matching", unit="jobs") as pbar:
            for i, job in enumerate(jobs_list):
                try:
                    result = self.create_comprehensive_match(resume_json, job)
                    results.append(result)
                    
                    pbar.update(1)
                    
                    # Memory cleanup every 100 jobs
                    if i % 100 == 0:
                        gc.collect()
                
                except Exception as e:
                    logger.warning(f"Error processing job {i}: {e}")
                    continue
        
        # Sort by overall score
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        logger.info(f"Completed structured matching for {len(results):,} jobs")
        return results
    
    def export_detailed_results(self, results: List[JobMatchResult], output_file: str = 'structured_job_matches.csv'):
        """
        Export detailed results with breakdown
        """
        logger.info(f"Exporting {len(results):,} structured results to {output_file}")
        
        data = []
        for result in results:
            breakdown = result.match_breakdown
            row = {
                'job_title': result.job_title,
                'company': result.company,
                'overall_score': result.match_score,
                'recommendation': result.recommendation,
                'experience_score': breakdown.experience_match,
                'skills_score': breakdown.skill_match,
                'education_score': breakdown.education_match,
                'semantic_score': breakdown.semantic_match,
                'salary_range': result.salary_range,
                'location': result.location,
                'job_url': result.job_url,
                'missing_requirements': '; '.join(breakdown.missing_requirements),
                'improvement_suggestions': '; '.join(breakdown.improvement_suggestions),
                'experience_details': breakdown.experience_details,
                'matched_required_skills': '; '.join(breakdown.skill_details['matched_required']),
                'required_skills_match_pct': breakdown.skill_details['required_skills_match'],
                'preferred_skills_match_pct': breakdown.skill_details['preferred_skills_match']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('overall_score', ascending=False)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Detailed results exported to {output_file}")
        return df
    
    # Helper methods for structured data extraction
    
    def calculate_total_experience(self, resume_json: Dict) -> float:
        """Calculate total years of professional experience"""
        total_years = 0.0
        current_year = datetime.now().year
        
        for exp in resume_json.get('experience', []):
            duration = exp.get('duration', '')
            years = self.parse_duration_to_years(duration, current_year)
            total_years += years
        
        return round(total_years, 1)
    
    def parse_duration_to_years(self, duration: str, current_year: int) -> float:
        """Parse duration string like 'Jun 2022 - Present' to years"""
        try:
            if 'Present' in duration or 'present' in duration:
                # Extract start date
                parts = duration.split(' - ')
                if len(parts) >= 1:
                    start_date = parts[0].strip()
                    start_year = self.extract_year_from_date(start_date)
                    if start_year:
                        return current_year - start_year
            else:
                # Parse start and end dates
                parts = duration.split(' - ')
                if len(parts) == 2:
                    start_year = self.extract_year_from_date(parts[0].strip())
                    end_year = self.extract_year_from_date(parts[1].strip())
                    if start_year and end_year:
                        return end_year - start_year
        except:
            pass
        
        # Default fallback
        return 1.0
    
    def extract_year_from_date(self, date_str: str) -> Optional[int]:
        """Extract year from date string like 'Jun 2022'"""
        # Look for 4-digit year
        year_match = re.search(r'\b(20\d{2})\b', date_str)
        if year_match:
            return int(year_match.group(1))
        return None
    
    def extract_all_skills(self, resume_json: Dict) -> List[str]:
        """Extract all skills from resume, flattening categories"""
        all_skills = []
        
        if resume_json.get('skills'):
            if isinstance(resume_json['skills'], list):
                all_skills.extend(resume_json['skills'])
            elif isinstance(resume_json['skills'], dict):
                for skill_category in resume_json['skills'].values():
                    if isinstance(skill_category, list):
                        all_skills.extend(skill_category)
        
        # Clean and deduplicate
        cleaned_skills = list(set(skill.strip() for skill in all_skills if skill.strip()))
        return cleaned_skills
    
    def extract_education_level(self, resume_json: Dict) -> str:
        """Extract highest education level"""
        education = resume_json.get('education', [])
        if not education:
            return "None"
        
        degrees = [edu.get('degree', '').lower() for edu in education]
        
        if any('phd' in deg or 'doctorate' in deg for deg in degrees):
            return "PhD"
        elif any('master' in deg or 'ms' in deg or 'ma' in deg for deg in degrees):
            return "Masters"
        elif any('bachelor' in deg or 'bs' in deg or 'ba' in deg for deg in degrees):
            return "Bachelors"
        elif any('associate' in deg for deg in degrees):
            return "Associates"
        else:
            return "Other"
    
    def extract_domain_experience(self, resume_json: Dict) -> List[str]:
        """Extract domain/industry experience"""
        domains = set()
        
        for exp in resume_json.get('experience', []):
            title = exp.get('title', '').lower()
            company = exp.get('company', '').lower()
            description = exp.get('description', '').lower()
            
            # Check for engineering
            if any(word in title + company + description for word in ['engineer', 'engineering']):
                domains.add('Engineering')
            
            # Check for research/science
            if any(word in title + company + description for word in ['research', 'scientist', 'r&d']):
                domains.add('Research')
            
            # Check for data/analytics
            if any(word in title + company + description for word in ['data', 'analytics', 'analysis']):
                domains.add('Data Science')
            
            # Check for defense/aerospace
            if any(word in company for word in ['defense', 'aerospace', 'raytheon', 'boeing', 'lockheed']):
                domains.add('Defense/Aerospace')
            
            # Check for software/tech
            if any(word in title + description for word in ['software', 'developer', 'programming']):
                domains.add('Software Development')
        
        return list(domains)
    
    def extract_technologies(self, resume_json: Dict) -> List[str]:
        """Extract specific technologies and tools"""
        technologies = set()
        
        # Common technology patterns
        tech_patterns = [
            r'\b(python|java|c\+\+|javascript|sql|r|matlab)\b',
            r'\b(aws|azure|gcp|docker|kubernetes|git)\b',
            r'\b(tensorflow|pytorch|scikit-learn|pandas|numpy)\b',
            r'\b(linux|windows|macos|unix)\b'
        ]
        
        # Check experience descriptions
        for exp in resume_json.get('experience', []):
            description = exp.get('description', '').lower()
            for pattern in tech_patterns:
                matches = re.findall(pattern, description, re.IGNORECASE)
                technologies.update(matches)
        
        # Add skills that look like technologies
        skills = resume_json.get('skills', [])
        if isinstance(skills, list):
            for skill in skills:
                if any(tech in skill.lower() for tech in ['python', 'sql', 'aws', 'docker', 'git', 'linux']):
                    technologies.add(skill)
        
        return list(technologies)
    
    def get_most_recent_role(self, resume_json: Dict) -> str:
        """Get most recent job title"""
        experience = resume_json.get('experience', [])
        if experience:
            return experience[0].get('title', 'Unknown')
        return 'Unknown'
    
    def parse_experience_requirement(self, experience_level: str) -> int:
        """Parse minimum years from experience level text"""
        level_lower = experience_level.lower()
        
        # Look for explicit numbers first
        number_match = re.search(r'(\d+)\s*\+?\s*years?', level_lower)
        if number_match:
            return int(number_match.group(1))
        
        # Map common terms to years
        if any(term in level_lower for term in ['entry', 'junior', 'new grad', '0-']):
            return 0
        elif any(term in level_lower for term in ['mid', 'intermediate', '3-', '2-3']):
            return 3
        elif any(term in level_lower for term in ['senior', '5+', '5-']):
            return 5
        elif any(term in level_lower for term in ['lead', 'principal', 'staff', '8+', '7+']):
            return 8
        elif any(term in level_lower for term in ['director', 'executive', '10+']):
            return 10
        else:
            return 2  # Default assumption
    
    def parse_education_requirements(self, education_list: List[str]) -> str:
        """Parse education requirements"""
        if not education_list:
            return "None"
        
        education_text = ' '.join(education_list).lower()
        
        if any(term in education_text for term in ['phd', 'doctorate', 'doctoral']):
            return "PhD"
        elif any(term in education_text for term in ['master', 'ms', 'ma', 'mba']):
            return "Masters"
        elif any(term in education_text for term in ['bachelor', 'bs', 'ba', 'degree']):
            return "Bachelors"
        else:
            return "None"
    
    def extract_job_technologies(self, job_json: Dict) -> List[str]:
        """Extract technologies mentioned in job description"""
        technologies = set()
        
        # Check all text in job posting
        all_text = ""
        if job_json.get('description', {}).get('summary'):
            all_text += job_json['description']['summary'].lower() + " "
        
        if job_json.get('requirements', {}).get('required_skills'):
            all_text += ' '.join(job_json['requirements']['required_skills']).lower() + " "
        
        # Common tech patterns
        tech_patterns = [
            r'\b(python|java|c\+\+|javascript|sql|r|matlab)\b',
            r'\b(aws|azure|gcp|docker|kubernetes|git)\b',
            r'\b(tensorflow|pytorch|scikit-learn|pandas|numpy)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            technologies.update(matches)
        
        return list(technologies)
    
    def determine_seniority_level(self, experience_level: str) -> str:
        """Determine seniority level from text"""
        level_lower = experience_level.lower()
        
        if any(term in level_lower for term in ['entry', 'junior', 'new grad']):
            return "Entry"
        elif any(term in level_lower for term in ['mid', 'intermediate']):
            return "Mid"
        elif any(term in level_lower for term in ['senior']):
            return "Senior"
        elif any(term in level_lower for term in ['lead', 'principal', 'staff']):
            return "Lead"
        elif any(term in level_lower for term in ['director', 'executive']):
            return "Executive"
        else:
            return "Unknown"

def main():
    """
    Main execution with structured matching
    """
    # Configuration
    BATCH_SIZE = 150
    MAX_JOBS = None  # Set to number for testing (e.g., 500)
    MIN_SCORE_FILTER = 40.0  # Only save jobs with 40%+ match
    
    # File paths
    resume_path = '/Users/jdtarriela/Documents/git/RoleRadar/src/jtarriela_resume.json'
    jobs_path = '/Users/jdtarriela/Documents/git/RoleRadar/src/processed_jobs/processed_netflix.json'
    output_path = '/Users/jdtarriela/Documents/git/RoleRadar/src/structured_job_matches.csv'
    
    logger.info("="*80)
    logger.info("🎯 STRUCTURED JOB MATCHING WITH FIELD ANALYSIS")
    logger.info("="*80)
    
    # Initialize structured matcher
    matcher = StructuredJobMatcher(batch_size=BATCH_SIZE)
    
    # Load data
    logger.info("Loading resume data...")
    with open(resume_path, 'r') as f:
        resume_data = json.load(f)
    
    logger.info("Loading jobs data...")
    with open(jobs_path, 'r') as f:
        jobs_data = json.load(f)
    
    if MAX_JOBS:
        jobs_data = jobs_data[:MAX_JOBS]
        logger.info(f"Limited to first {MAX_JOBS:,} jobs for testing")
    
    logger.info(f"Loaded {len(jobs_data):,} jobs to process")
    
    # Process with structured matching
    results = matcher.process_job_batch_structured(resume_data, jobs_data)
    
    # Filter results
    filtered_results = [r for r in results if r.match_score >= MIN_SCORE_FILTER]
    logger.info(f"Filtered to {len(filtered_results):,} jobs with score >= {MIN_SCORE_FILTER}%")
    
    # Export detailed results
    df = matcher.export_detailed_results(filtered_results, output_path)
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 STRUCTURED JOB MATCHING COMPLETE")
    print("="*80)
    
    # Statistics
    total_jobs = len(results)
    excellent_matches = len([r for r in filtered_results if r.match_score >= 80])
    strong_matches = len([r for r in filtered_results if 70 <= r.match_score < 80])
    good_matches = len([r for r in filtered_results if 60 <= r.match_score < 70])
    potential_matches = len([r for r in filtered_results if 50 <= r.match_score < 60])
    
    print(f"📊 STRUCTURED MATCH BREAKDOWN:")
    print(f"   • Total Jobs Analyzed: {total_jobs:,}")
    print(f"   • Jobs Above {MIN_SCORE_FILTER}% threshold: {len(filtered_results):,}")
    print(f"   • 🎯 Excellent Matches (80%+): {excellent_matches}")
    print(f"   • 🚀 Strong Matches (70-80%): {strong_matches}")
    print(f"   • ✅ Good Matches (60-70%): {good_matches}")
    print(f"   • 🤔 Potential Matches (50-60%): {potential_matches}")
    
    print(f"\n💾 OUTPUT:")
    print(f"   • Detailed results: {output_path}")
    print(f"   • Includes skill breakdowns, missing requirements, suggestions")
    
    # Show top 5 matches with details
    print(f"\n🏆 TOP 5 STRUCTURED MATCHES:")
    print("-" * 70)
    for i, result in enumerate(filtered_results[:5], 1):
        breakdown = result.match_breakdown
        print(f"\n{i}. {result.job_title} at {result.company}")
        print(f"   Overall Score: {result.match_score}%")
        print(f"   └── Experience: {breakdown.experience_match:.1f}% | Skills: {breakdown.skill_match:.1f}% | Education: {breakdown.education_match:.1f}% | Semantic: {breakdown.semantic_match:.1f}%")
        print(f"   {result.recommendation}")
        print(f"   Salary: {result.salary_range} | Location: {result.location}")
        if breakdown.missing_requirements:
            print(f"   ⚠️  Missing: {', '.join(breakdown.missing_requirements[:3])}")
        if breakdown.improvement_suggestions:
            print(f"   💡 Suggestions: {breakdown.improvement_suggestions[0]}")
    
    print(f"\n✅ Structured matching complete! Check CSV for detailed breakdowns.")

if __name__ == "__main__":
    main()