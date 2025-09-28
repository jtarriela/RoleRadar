#!/usr/bin/env python3
"""
Simple Job Matcher using Gemini Embeddings + Cosine Similarity
"""

import json
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dataclasses import dataclass
import logging
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class JobMatch:
    """Simple job match result"""
    job_title: str
    company: str
    match_score: float
    job_url: str
    salary_range: str
    location: str
    recommendation: str

class GeminiJobMatcher:
    """
    Clean job matcher using Gemini embeddings - simple and effective
    """
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 50):
        """
        Initialize Gemini job matcher
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            batch_size: Number of jobs to process per batch
        """
        # Configure Gemini API
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.\n"
                "Get your key at: https://aistudio.google.com/app/apikey"
            )
        
        genai.configure(api_key=api_key)
        self.batch_size = batch_size
        
        # Cache for resume embedding (compute once, reuse)
        self.resume_embedding_cache = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("✅ Gemini API configured successfully")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding from Gemini API with rate limiting
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        try:
            result = genai.embed_content(
                model="gemini-embedding-001",  # Latest Gemini embedding model
                content=text,
                task_type="semantic_similarity"    # Optimized for similarity tasks
            )
            
            self.last_request_time = time.time()
            
            # Extract embedding vector
            embedding = np.array(result['embedding'])
            return embedding
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            # Fallback: return zero vector (will result in 0% match)
            return np.zeros(768)  # Gemini embeddings are 768-dimensional
    
    def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts with batching
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process with progress bar
        for text in tqdm(texts, desc="Getting Gemini embeddings", unit="text"):
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
            
            # Small delay to be nice to the API
            time.sleep(0.05)
        
        return embeddings
    
    def format_resume_text(self, resume_json: Dict) -> str:
        """
        Convert resume JSON to optimized text for embedding
        """
        sections = []
        
        # Professional summary
        if resume_json.get('summary'):
            sections.append(f"Professional Summary: {resume_json['summary']}")
        
        # Current roles
        if resume_json.get('roles'):
            role_titles = [role['title'] for role in resume_json['roles']]
            sections.append(f"Current Roles: {', '.join(role_titles)}")
        
        # Professional experience (detailed but not overwhelming)
        if resume_json.get('experience'):
            exp_texts = []
            for exp in resume_json['experience'][:3]:  # Top 3 most recent
                # Truncate very long descriptions
                desc = exp['description']
                if len(desc) > 600:
                    desc = desc[:600] + "..."
                
                exp_detail = f"{exp['title']} at {exp['company']} ({exp['duration']}): {desc}"
                exp_texts.append(exp_detail)
            
            sections.append(f"Professional Experience:\n{chr(10).join(exp_texts)}")
        
        # Skills (all skills, but organized)
        if resume_json.get('skills'):
            if isinstance(resume_json['skills'], list):
                skills_text = ', '.join(resume_json['skills'])
            else:
                # Flatten all skill categories
                all_skills = []
                if isinstance(resume_json['skills'], dict):
                    for skill_category in resume_json['skills'].values():
                        if isinstance(skill_category, list):
                            all_skills.extend(skill_category)
                elif isinstance(resume_json['skills'], list):
                    all_skills.extend(resume_json['skills'])
                skills_text = ', '.join(all_skills)
            
            sections.append(f"Technical Skills: {skills_text}")
        
        # Education
        if resume_json.get('education'):
            edu_texts = []
            for edu in resume_json['education']:
                edu_detail = f"{edu.get('degree', '')} in {edu.get('field', '')} ({edu.get('year', '')})"
                edu_texts.append(edu_detail)
            sections.append(f"Education: {', '.join(edu_texts)}")
        
        # Certifications
        if resume_json.get('certifications'):
            sections.append(f"Certifications: {', '.join(resume_json['certifications'])}")
        
        formatted_text = "\n\n".join(sections)
        
        # Ensure text isn't too long for API (Gemini has limits)
        max_length = 8000  # Conservative limit
        if len(formatted_text) > max_length:
            formatted_text = formatted_text[:max_length] + "..."
        
        return formatted_text
    
    def format_job_text(self, job_json: Dict) -> str:
        """
        Convert job JSON to optimized text for embedding
        """
        sections = []
        
        # Job title and company
        if job_json.get('job_info'):
            job_info = job_json['job_info']
            sections.append(f"Position: {job_info.get('title', '')} at {job_info.get('company', '')}")
            
            if job_info.get('location'):
                sections.append(f"Location: {job_info['location']}")
        
        # Job summary
        if job_json.get('description', {}).get('summary'):
            summary = job_json['description']['summary']
            # Truncate if too long
            if len(summary) > 800:
                summary = summary[:800] + "..."
            sections.append(f"Role Summary: {summary}")
        
        # Requirements
        if job_json.get('requirements'):
            req = job_json['requirements']
            
            if req.get('experience_level'):
                sections.append(f"Experience Level: {req['experience_level']}")
            
            if req.get('required_skills'):
                req_skills = ', '.join(req['required_skills'])
                # Truncate if too long
                if len(req_skills) > 500:
                    req_skills = req_skills[:500] + "..."
                sections.append(f"Required Skills: {req_skills}")
            
            if req.get('preferred_skills'):
                pref_skills = ', '.join(req['preferred_skills'])
                if len(pref_skills) > 300:
                    pref_skills = pref_skills[:300] + "..."
                sections.append(f"Preferred Skills: {pref_skills}")
            
            if req.get('education'):
                sections.append(f"Education Requirements: {', '.join(req['education'])}")
        
        # Responsibilities
        if job_json.get('description', {}).get('responsibilities'):
            resp_list = job_json['description']['responsibilities']
            resp_text = ' '.join(resp_list[:5])  # Top 5 responsibilities
            if len(resp_text) > 600:
                resp_text = resp_text[:600] + "..."
            sections.append(f"Key Responsibilities: {resp_text}")
        
        # Keywords for additional context
        if job_json.get('description', {}).get('keywords'):
            keywords = ', '.join(job_json['description']['keywords'][:15])  # Top 15 keywords
            sections.append(f"Key Technologies: {keywords}")
        
        formatted_text = "\n\n".join(sections)
        
        # Ensure text isn't too long
        max_length = 6000
        if len(formatted_text) > max_length:
            formatted_text = formatted_text[:max_length] + "..."
        
        return formatted_text
    
    def get_resume_embedding(self, resume_json: Dict) -> np.ndarray:
        """
        Get cached resume embedding or compute if not cached
        """
        resume_id = resume_json.get('full_name', 'default')
        
        if resume_id not in self.resume_embedding_cache:
            logger.info(f"Computing Gemini embedding for resume: {resume_id}")
            resume_text = self.format_resume_text(resume_json)
            resume_embedding = self.get_embedding(resume_text)
            
            self.resume_embedding_cache[resume_id] = {
                'text': resume_text,
                'embedding': resume_embedding
            }
            logger.info(f"✅ Resume embedding cached for {resume_id}")
        
        return self.resume_embedding_cache[resume_id]['embedding']
    
    def calculate_job_matches(self, resume_json: Dict, jobs_list: List[Dict]) -> List[JobMatch]:
        """
        Calculate matches for all jobs using Gemini embeddings
        """
        logger.info(f"🚀 Starting Gemini embedding matching for {len(jobs_list):,} jobs")
        
        # Get resume embedding (cached)
        resume_embedding = self.get_resume_embedding(resume_json)
        
        # Format all job texts
        logger.info("📝 Formatting job descriptions...")
        job_texts = []
        for job in tqdm(jobs_list, desc="Formatting jobs", unit="job"):
            job_text = self.format_job_text(job)
            job_texts.append(job_text)
        
        # Get job embeddings from Gemini
        logger.info("🧠 Getting Gemini embeddings for jobs...")
        job_embeddings = self.get_batch_embeddings(job_texts)
        
        # Calculate cosine similarities
        logger.info("📊 Calculating cosine similarities...")
        results = []
        
        for i, (job, job_embedding) in enumerate(zip(jobs_list, job_embeddings)):
            try:
                # Calculate cosine similarity
                similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
                match_score = max(0, min(100, similarity * 100))  # Convert to percentage
                
                # Generate recommendation
                recommendation = self.get_recommendation(match_score)
                
                # Extract job info
                job_info = job.get('job_info', {})
                
                result = JobMatch(
                    job_title=job_info.get('title', 'Unknown'),
                    company=job_info.get('company', 'Unknown'),
                    match_score=round(match_score, 2),
                    job_url=job_info.get('job_url', ''),
                    salary_range=job_info.get('salary_range', 'Not specified'),
                    location=job_info.get('location', 'Not specified'),
                    recommendation=recommendation
                )
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing job {i}: {e}")
                continue
        
        # Sort by match score
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        logger.info(f"✅ Completed Gemini matching for {len(results):,} jobs")
        return results
    
    def get_recommendation(self, match_score: float) -> str:
        """
        Generate recommendation based on match score
        """
        if match_score >= 85:
            return "🎯 EXCELLENT MATCH - Apply immediately!"
        elif match_score >= 75:
            return "🚀 STRONG MATCH - Highly recommended"
        elif match_score >= 65:
            return "✅ GOOD MATCH - Consider applying"
        elif match_score >= 55:
            return "🤔 POTENTIAL MATCH - Worth reviewing"
        elif match_score >= 45:
            return "📝 WEAK MATCH - Consider if desperate"
        else:
            return "❌ LOW MATCH - Skip this one"
    
    def export_results(self, results: List[JobMatch], output_file: str = 'gemini_job_matches.csv') -> pd.DataFrame:
        """
        Export results to CSV
        """
        logger.info(f"💾 Exporting {len(results):,} results to {output_file}")
        
        data = []
        for result in results:
            row = {
                'job_title': result.job_title,
                'company': result.company,
                'match_score': result.match_score,
                'recommendation': result.recommendation,
                'salary_range': result.salary_range,
                'location': result.location,
                'job_url': result.job_url
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('match_score', ascending=False)
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Results exported to {output_file}")
        return df
    
    def filter_results(self, results: List[JobMatch], min_score: float = 50.0) -> List[JobMatch]:
        """
        Filter results by minimum score
        """
        filtered = [r for r in results if r.match_score >= min_score]
        logger.info(f"🔍 Filtered to {len(filtered):,} jobs with score >= {min_score}%")
        return filtered

def estimate_cost(num_jobs: int) -> Dict[str, float]:
    """
    Estimate Gemini API costs
    """
    # Gemini text-embedding-004 pricing (as of 2024)
    # Free tier: 1,500 requests per day
    # Paid: $0.000025 per 1K characters
    
    avg_chars_per_job = 3000  # Conservative estimate
    total_chars = num_jobs * avg_chars_per_job
    
    # Add resume characters (computed once)
    resume_chars = 2000
    total_chars += resume_chars
    
    # Calculate cost
    cost_per_1k_chars = 0.000025
    estimated_cost = (total_chars / 1000) * cost_per_1k_chars
    
    return {
        'total_characters': total_chars,
        'estimated_cost_usd': estimated_cost,
        'free_tier_sufficient': num_jobs <= 1500  # Rough estimate
    }

def main():
    """
    Main execution function using Gemini embeddings
    """
    # Configuration
    MAX_JOBS = None  # Set to number for testing (e.g., 100)
    MIN_SCORE_FILTER = 45.0  # Only save jobs with 45%+ match
    BATCH_SIZE = 50  # Gemini rate limits
    
    # File paths
    resume_path = '../runtime_data/processed_resumes/jtarriela_resume.json'
    jobs_path = '../runtime_data/processed_job_data/processed_netflix.json'
    output_path = '../runtime_data/match_results/netflix_gemini_job_matches.csv'
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("📋 To fix this:")
        print("   1. Get your API key: https://aistudio.google.com/app/apikey")
        print("   2. Set environment variable:")
        print("      export GEMINI_API_KEY='your-api-key-here'")
        print("   3. Or add to your shell profile (.bashrc, .zshrc, etc.)")
        return
    
    print("=" * 80)
    print("🤖 GEMINI EMBEDDINGS JOB MATCHER")
    print("=" * 80)
    
    # Load data
    logger.info("📂 Loading data...")
    with open(resume_path, 'r') as f:
        resume_data = json.load(f)
    
    with open(jobs_path, 'r') as f:
        jobs_data = json.load(f)
    
    # Limit for testing if specified
    if MAX_JOBS:
        jobs_data = jobs_data[:MAX_JOBS]
        logger.info(f"🔧 Limited to first {MAX_JOBS:,} jobs for testing")
    
    total_jobs = len(jobs_data)
    logger.info(f"📊 Loaded {total_jobs:,} jobs to process")
    
    # Estimate costs
    cost_info = estimate_cost(total_jobs)
    print(f"\n💰 COST ESTIMATE:")
    print(f"   • Total characters: {cost_info['total_characters']:,}")
    print(f"   • Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")
    print(f"   • Free tier sufficient: {'Yes' if cost_info['free_tier_sufficient'] else 'No'}")
    
    if cost_info['estimated_cost_usd'] > 1.0:
        response = input(f"\n⚠️  Estimated cost is ${cost_info['estimated_cost_usd']:.2f}. Continue? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled by user")
            return
    
    # Initialize matcher
    matcher = GeminiJobMatcher(batch_size=BATCH_SIZE)
    
    # Process jobs
    start_time = time.time()
    results = matcher.calculate_job_matches(resume_data, jobs_data)
    end_time = time.time()
    
    # Filter results
    filtered_results = matcher.filter_results(results, MIN_SCORE_FILTER)
    
    # Export results
    df = matcher.export_results(filtered_results, output_path)
    
    # Performance summary
    processing_time = end_time - start_time
    jobs_per_minute = (total_jobs / processing_time) * 60 if processing_time > 0 else 0
    
    print("\n" + "=" * 80)
    print("🎯 GEMINI JOB MATCHING COMPLETE")
    print("=" * 80)
    
    # Statistics
    excellent_matches = len([r for r in filtered_results if r.match_score >= 85])
    strong_matches = len([r for r in filtered_results if 75 <= r.match_score < 85])
    good_matches = len([r for r in filtered_results if 65 <= r.match_score < 75])
    potential_matches = len([r for r in filtered_results if 55 <= r.match_score < 65])
    
    print(f"📊 PERFORMANCE METRICS:")
    print(f"   • Total Jobs Processed: {total_jobs:,}")
    print(f"   • Processing Time: {processing_time/60:.1f} minutes")
    print(f"   • Speed: {jobs_per_minute:.1f} jobs/minute")
    print(f"   • Jobs Above {MIN_SCORE_FILTER}% threshold: {len(filtered_results):,}")
    
    print(f"\n🎯 MATCH QUALITY BREAKDOWN:")
    print(f"   • 🎯 Excellent Matches (85%+): {excellent_matches}")
    print(f"   • 🚀 Strong Matches (75-85%): {strong_matches}")
    print(f"   • ✅ Good Matches (65-75%): {good_matches}")
    print(f"   • 🤔 Potential Matches (55-65%): {potential_matches}")
    
    print(f"\n💾 OUTPUT:")
    print(f"   • Results saved to: {output_path}")
    print(f"   • Open in Excel/Sheets for review")
    
    # Show top 10 matches
    print(f"\n🏆 TOP 10 GEMINI MATCHES:")
    print("-" * 70)
    for i, result in enumerate(filtered_results[:10], 1):
        print(f"{i:2d}. {result.job_title} at {result.company}")
        print(f"     Score: {result.match_score}% | {result.recommendation}")
        print(f"     Salary: {result.salary_range} | Location: {result.location}")
        print(f"     🔗 {result.job_url}")
        print()
    
    print("✅ Gemini embedding matching complete!")
    
    # Cost summary
    if processing_time > 0:
        actual_cost = cost_info['estimated_cost_usd']
        print(f"\n💳 ACTUAL COST: ~${actual_cost:.4f}")

if __name__ == "__main__":
    main()