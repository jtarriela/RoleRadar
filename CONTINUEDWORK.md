# RoleRadar: AI-Powered Job Matching Platform

Transform your existing job scraper into a full-stack web application that intelligently matches resumes to job postings.

## 🎯 Project Overview

**Current State**: Collection of Python scripts that scrape jobs, parse resumes, and generate matches  
**Target State**: Production web application with database, API, and frontend interface

## 📁 Repository Structure

```
roleradar/
├── README.md                    # This file
├── docker-compose.yml           # Full application stack
├── requirements.txt             # Python dependencies
│
├── database/
│   ├── init.sql                # Database schema creation
│   ├── models.py               # SQLAlchemy models
│   └── connection.py           # Database connection
│
├── src/                        # Your existing ML pipeline
│   ├── resume_parser.py        # ✅ Already exists
│   ├── jobdata_markdown_to_json.py  # ✅ Already exists  
│   ├── structured_text_mapping.py   # ✅ Already exists
│   ├── sitemap_parse.py        # ✅ Already exists
│   └── db/
│       ├── crud.py             # Database operations
│       └── models.py           # Data models
│
├── cli/
│   └── admin.py                # Command-line interface
│
├── api/
│   ├── main.py                 # FastAPI application
│   ├── routes/                 # API endpoints
│   └── middleware.py           # CORS, auth, etc.
│
├── frontend/                   # React/Next.js application
│   ├── pages/                  # Application pages
│   ├── components/             # Reusable components
│   └── services/               # API integration
│
└── deploy/
    ├── Dockerfile              # Container definition
    ├── nginx.conf              # Reverse proxy config
    └── aws/                    # Deployment scripts
```

---

## ⚡ Key Features: Smart Keyword Filtering

### Speed & Cost Optimization
Instead of running expensive LLM analysis on all jobs, users can pre-filter by keywords:

**Example Workflow:**
```
1. User uploads resume → Parsed with your existing resume_parser.py
2. User enters keywords: "python, react, aws, remote"
3. System filters 50,000 jobs → 2,500 matching jobs (95% reduction!)
4. LLM analysis runs on 2,500 jobs instead of 50,000
5. Results displayed in sortable table with direct application links
```

**Benefits:**
- **95% faster processing** - Only analyze relevant jobs
- **95% cheaper LLM costs** - Fewer API calls needed  
- **Better matches** - Focus on jobs user actually wants
- **Real-time feedback** - Shows job count as user types keywords

### Frontend User Experience

**Single Page Workflow:**
1. **Upload Resume** - Drag-drop with instant parsing feedback
2. **Filter Jobs** - Keyword input with live job count updates
3. **View Results** - Sortable table with company, title, salary, match %
4. **Apply** - Direct links to company application pages

**Smart Features:**
- **Keyword Suggestions** - Shows popular skills from your job database
- **Live Filtering** - Job count updates as user types
- **Match Score Color Coding** - Green (70%+ auto-apply), Yellow (50-69% improve resume), Red (review)
- **Export Options** - CSV download, bulk apply to top matches
- **Cost Transparency** - Shows user how much processing time they're saving

### Example Results Table

| Match % | Company | Job Title | Salary | Location | Action |
|---------|---------|-----------|---------|----------|---------|
| 87% 🟢 | Netflix | Senior Python Engineer | $150k-200k | Remote | [Apply →](https://netflix.com/jobs/123) |
| 75% 🟢 | Google | Full Stack Developer | $140k-180k | Mountain View | [Apply →](https://google.com/jobs/456) |
| 62% 🟡 | Meta | React Developer | $130k-170k | Menlo Park | [Apply →](https://meta.com/jobs/789) |

### Backend Intelligence
Your existing ML pipeline runs exactly the same, just on a filtered subset:
- **Resume Parser** - Same `parse_resume()` function
- **Job Matching** - Same `StructuredJobMatcher` algorithm  
- **Database Storage** - Same schema, just more efficient queries

---

## 🏗️ Phase 1: Database Integration

### Step 1.1: Database Schema Design

Based on your existing JSON structures, here's the PostgreSQL schema:

```sql
-- database/init.sql

-- Companies being scraped
CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    careers_url TEXT,
    sitemap_url TEXT,
    last_scraped TIMESTAMPTZ,
    status TEXT DEFAULT 'active', -- active, paused, error
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Job postings (extends your existing JOBS table concept)
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    location TEXT,
    seniority TEXT, -- junior, mid, senior, executive
    employment_type TEXT, -- full-time, contract, etc.
    salary_min INTEGER,
    salary_max INTEGER,
    
    -- Your existing ML processing results
    parsed_data JSONB NOT NULL, -- Full job JSON from jobdata_markdown_to_json.py
    skills_extracted TEXT[], -- Array of extracted skills
    requirements_summary TEXT,
    
    -- Full-text search (your existing approach)
    jd_tsv TSVECTOR, -- Full-text search vector
    
    -- Metadata
    scraped_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    scrape_fingerprint TEXT, -- Hash to detect changes
    
    -- Indexes
    CONSTRAINT jobs_url_unique UNIQUE(url)
);

-- Create GIN index for full-text search
CREATE INDEX idx_jobs_fts ON jobs USING GIN(jd_tsv);
CREATE INDEX idx_jobs_company ON jobs(company_id);
CREATE INDEX idx_jobs_scraped ON jobs(scraped_at);

-- Resumes (from your resume_parser.py output)
CREATE TABLE resumes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID, -- For future user system
    filename TEXT NOT NULL,
    
    -- Your existing resume JSON structure
    parsed_data JSONB NOT NULL, -- Output from parse_resume()
    
    -- Extracted fields for easy querying
    candidate_name TEXT,
    email TEXT,
    phone TEXT,
    location TEXT,
    experience_years INTEGER,
    skills_extracted TEXT[],
    
    -- Metadata
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    version INTEGER DEFAULT 1
);

-- Job matches (from your structured_text_mapping.py)
CREATE TABLE job_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resume_id UUID REFERENCES resumes(id),
    job_id UUID REFERENCES jobs(id),
    
    -- Your existing match scoring
    match_score DECIMAL(5,2) NOT NULL, -- 0.00 to 100.00
    skill_match_score DECIMAL(5,2),
    experience_match_score DECIMAL(5,2),
    location_match_score DECIMAL(5,2),
    
    -- Match analysis
    matching_skills TEXT[],
    missing_skills TEXT[],
    recommendations TEXT, -- LLM suggestions for resume improvement
    
    -- Status tracking
    status TEXT DEFAULT 'pending', -- pending, auto_apply, improve_resume, not_suitable
    applied_at TIMESTAMPTZ,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prevent duplicate matches
    CONSTRAINT unique_resume_job UNIQUE(resume_id, job_id)
);

-- Scraping jobs tracking
CREATE TABLE scraping_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id),
    task_type TEXT NOT NULL, -- sitemap_extract, crawl_jobs, parse_jobs
    status TEXT DEFAULT 'queued', -- queued, running, completed, failed
    
    -- Progress tracking
    urls_found INTEGER DEFAULT 0,
    urls_processed INTEGER DEFAULT 0,
    jobs_extracted INTEGER DEFAULT 0,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Step 1.2: Database Models

```python
# src/db/models.py
from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean, ARRAY, JSON, ForeignKey, DECIMAL
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    careers_url = Column(String)
    sitemap_url = Column(String)
    last_scraped = Column(DateTime(timezone=True))
    status = Column(String, default='active')
    created_at = Column(DateTime(timezone=True), server_default='NOW()')
    
    # Relationships
    jobs = relationship("Job", back_populates="company")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'))
    url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    location = Column(String)
    seniority = Column(String)
    employment_type = Column(String)
    salary_min = Column(Integer)
    salary_max = Column(Integer)
    
    # Your ML processing results
    parsed_data = Column(JSON, nullable=False)  # From jobdata_markdown_to_json.py
    skills_extracted = Column(ARRAY(String))
    requirements_summary = Column(Text)
    jd_tsv = Column(TSVECTOR)  # Full-text search
    
    # Metadata
    scraped_at = Column(DateTime(timezone=True), server_default='NOW()')
    updated_at = Column(DateTime(timezone=True), server_default='NOW()')
    scrape_fingerprint = Column(String)
    
    # Relationships
    company = relationship("Company", back_populates="jobs")
    matches = relationship("JobMatch", back_populates="job")

class Resume(Base):
    __tablename__ = "resumes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True))
    filename = Column(String, nullable=False)
    
    # Your resume parser output
    parsed_data = Column(JSON, nullable=False)  # From parse_resume()
    
    # Extracted for easy querying
    candidate_name = Column(String)
    email = Column(String)
    phone = Column(String)
    location = Column(String)
    experience_years = Column(Integer)
    skills_extracted = Column(ARRAY(String))
    
    # Metadata
    uploaded_at = Column(DateTime(timezone=True), server_default='NOW()')
    is_active = Column(Boolean, default=True)
    version = Column(Integer, default=1)
    
    # Relationships
    matches = relationship("JobMatch", back_populates="resume")

class JobMatch(Base):
    __tablename__ = "job_matches"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resume_id = Column(UUID(as_uuid=True), ForeignKey('resumes.id'))
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'))
    
    # Your match scoring from structured_text_mapping.py
    match_score = Column(DECIMAL(5,2), nullable=False)
    skill_match_score = Column(DECIMAL(5,2))
    experience_match_score = Column(DECIMAL(5,2))
    location_match_score = Column(DECIMAL(5,2))
    
    # Analysis
    matching_skills = Column(ARRAY(String))
    missing_skills = Column(ARRAY(String))
    recommendations = Column(Text)
    
    # Status
    status = Column(String, default='pending')
    applied_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default='NOW()')
    
    # Relationships
    resume = relationship("Resume", back_populates="matches")
    job = relationship("Job", back_populates="matches")
```

### Step 1.3: Database Operations

```python
# src/db/crud.py
from sqlalchemy.orm import Session
from . import models
from typing import List, Optional
import json

class ResumeOperations:
    @staticmethod
    def create_resume(db: Session, filename: str, parsed_data: dict) -> models.Resume:
        """Save parsed resume to database (replaces your save_resume_json)"""
        resume = models.Resume(
            filename=filename,
            parsed_data=parsed_data,
            candidate_name=parsed_data.get('personal_info', {}).get('name'),
            email=parsed_data.get('personal_info', {}).get('email'),
            skills_extracted=parsed_data.get('skills', {}).get('technical', [])
        )
        db.add(resume)
        db.commit()
        db.refresh(resume)
        return resume
    
    @staticmethod
    def get_resume(db: Session, resume_id: str) -> Optional[models.Resume]:
        return db.query(models.Resume).filter(models.Resume.id == resume_id).first()

class JobOperations:
    @staticmethod
    def create_job(db: Session, company_id: str, url: str, parsed_data: dict) -> models.Job:
        """Save scraped job to database (replaces JSON file storage)"""
        job_info = parsed_data.get('job_info', {})
        
        job = models.Job(
            company_id=company_id,
            url=url,
            title=job_info.get('title'),
            location=job_info.get('location'),
            seniority=job_info.get('seniority_level'),
            employment_type=job_info.get('employment_type'),
            parsed_data=parsed_data,
            skills_extracted=job_info.get('required_skills', [])
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    
    @staticmethod
    def get_jobs_by_company(db: Session, company_name: str) -> List[models.Job]:
        return db.query(models.Job).join(models.Company).filter(
            models.Company.name == company_name
        ).all()

class MatchOperations:
    @staticmethod
    def create_match(db: Session, resume_id: str, job_id: str, match_data: dict) -> models.JobMatch:
        """Save match results (replaces CSV output)"""
        match = models.JobMatch(
            resume_id=resume_id,
            job_id=job_id,
            match_score=match_data['match_score'],
            matching_skills=match_data.get('matching_skills', []),
            missing_skills=match_data.get('missing_skills', []),
            recommendations=match_data.get('recommendations'),
            status='auto_apply' if match_data['match_score'] >= 70 else 
                   'improve_resume' if match_data['match_score'] >= 50 else 'not_suitable'
        )
        db.add(match)
        db.commit()
        db.refresh(match)
        return match
```

### Step 1.4: CLI Administration Tool

```python
# cli/admin.py
import click
import asyncio
from sqlalchemy.orm import Session
from src.db.connection import get_db
from src.db import crud
from src.resume_parser import parse_resume
from src.jobdata_markdown_to_json import JobProcessor
from src.structured_text_mapping import StructuredJobMatcher
from src.sitemap_parse import extract_sitemap_urls
import json

@click.group()
def cli():
    """RoleRadar Admin CLI - Manage your job matching pipeline"""
    pass

@cli.command()
@click.option('--name', required=True, help='Company name')
@click.option('--sitemap-url', required=True, help='Company sitemap URL')
def add_company(name: str, sitemap_url: str):
    """Add a company to scrape"""
    db = next(get_db())
    company = crud.CompanyOperations.create_company(
        db, name=name, sitemap_url=sitemap_url
    )
    click.echo(f"✅ Added company: {company.name} (ID: {company.id})")

@cli.command()
@click.option('--company', help='Company name to scrape (default: all)')
@click.option('--max-jobs', type=int, help='Limit number of jobs to process')
def scrape(company: str, max_jobs: int):
    """Scrape jobs from company websites"""
    db = next(get_db())
    
    if company:
        companies = [crud.CompanyOperations.get_by_name(db, company)]
    else:
        companies = crud.CompanyOperations.get_all_active(db)
    
    for comp in companies:
        click.echo(f"🕷️  Scraping {comp.name}...")
        
        # Use your existing sitemap extraction
        urls = extract_sitemap_urls(comp.sitemap_url)
        click.echo(f"   Found {len(urls)} job URLs")
        
        # Use your existing job processor  
        processor = JobProcessor()
        job_count = 0
        
        for url in urls[:max_jobs] if max_jobs else urls:
            try:
                # Your crawl4ai + LLM processing pipeline
                job_data = processor.process_single_job(url)
                
                # Save to database instead of JSON file
                crud.JobOperations.create_job(
                    db, company_id=comp.id, url=url, parsed_data=job_data
                )
                job_count += 1
                
                if job_count % 10 == 0:
                    click.echo(f"   Processed {job_count} jobs...")
                    
            except Exception as e:
                click.echo(f"   ❌ Failed to process {url}: {e}")
        
        click.echo(f"✅ Scraped {job_count} jobs for {comp.name}")

@cli.command()
@click.argument('resume_file', type=click.Path(exists=True))
def parse_resume(resume_file: str):
    """Parse a resume and save to database"""
    db = next(get_db())
    
    click.echo(f"📄 Parsing resume: {resume_file}")
    
    # Use your existing resume parser
    parsed_data = parse_resume(resume_file, use_llm=True)
    
    # Save to database instead of JSON file
    resume = crud.ResumeOperations.create_resume(
        db, filename=resume_file, parsed_data=parsed_data
    )
    
    click.echo(f"✅ Parsed resume saved with ID: {resume.id}")
    click.echo(f"   Candidate: {resume.candidate_name}")
    click.echo(f"   Skills: {len(resume.skills_extracted)} extracted")

@cli.command()
@click.option('--resume-id', required=True, help='Resume ID to match')
@click.option('--keywords', help='Comma-separated keywords to filter jobs (e.g., "python,react,aws")')
@click.option('--company', help='Limit to specific company')
@click.option('--min-score', type=float, default=50.0, help='Minimum match score')
@click.option('--max-jobs', type=int, default=1000, help='Maximum jobs to process')
def match(resume_id: str, keywords: str, company: str, min_score: float, max_jobs: int):
    """Run job matching for a resume with optional keyword filtering"""
    db = next(get_db())
    
    # Get resume
    resume = crud.ResumeOperations.get_resume(db, resume_id)
    if not resume:
        click.echo(f"❌ Resume {resume_id} not found")
        return
    
    # Parse keywords if provided
    keyword_list = []
    if keywords:
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
        click.echo(f"🔍 Filtering jobs by keywords: {', '.join(keyword_list)}")
    
    # Get jobs to match against (filtered or all)
    if keyword_list:
        jobs = crud.JobOperations.get_jobs_with_keywords(db, keyword_list, limit=max_jobs)
        total_jobs = crud.JobOperations.count_all(db)
        click.echo(f"   Found {len(jobs):,} jobs matching keywords (vs {total_jobs:,} total)")
        click.echo(f"   💰 Keyword filtering saves ~{((total_jobs - len(jobs)) / total_jobs * 100):.1f}% LLM API costs!")
    elif company:
        jobs = crud.JobOperations.get_jobs_by_company(db, company)
        click.echo(f"   Found {len(jobs):,} jobs at {company}")
    else:
        jobs = crud.JobOperations.get_all_active(db, limit=max_jobs)
        click.echo(f"   Processing {len(jobs):,} jobs (use --keywords to filter for speed)")
    
    if not jobs:
        click.echo("❌ No jobs found matching criteria")
        return
    
    click.echo(f"🎯 Running AI matching against {len(jobs):,} jobs...")
    
    # Use your existing structured matcher
    matcher = StructuredJobMatcher()
    
    match_count = 0
    auto_apply_count = 0
    improve_resume_count = 0
    
    with click.progressbar(jobs, label='Matching jobs') as job_bar:
        for job in job_bar:
            try:
                # Your existing matching logic
                match_result = matcher.calculate_match(
                    resume.parsed_data, job.parsed_data
                )
                
                if match_result['match_score'] >= min_score:
                    # Determine status based on your thresholds
                    if match_result['match_score'] >= 70:
                        status = 'auto_apply'
                        auto_apply_count += 1
                    elif match_result['match_score'] >= 50:
                        status = 'improve_resume' 
                        improve_resume_count += 1
                    else:
                        status = 'review'
                    
                    # Save to database instead of CSV
                    crud.MatchOperations.create_match(
                        db, 
                        resume_id=resume.id, 
                        job_id=job.id, 
                        match_data={
                            **match_result,
                            'status': status
                        }
                    )
                    match_count += 1
                    
            except Exception as e:
                click.echo(f"   ❌ Failed to match job {job.id}: {e}")
    
    click.echo(f"\n✅ Matching Complete!")
    click.echo(f"   📊 Created {match_count:,} matches above {min_score}% score")
    click.echo(f"   🚀 Auto-apply recommended: {auto_apply_count:,} jobs (≥70% match)")
    click.echo(f"   📝 Resume improvement suggested: {improve_resume_count:,} jobs (50-69% match)")
    
    if keyword_list:
        click.echo(f"   ⚡ Keyword filtering processed {len(jobs):,} jobs instead of {crud.JobOperations.count_all(db):,}")
        click.echo(f"   💡 Add more keywords to further narrow results, or remove --keywords to match all jobs")

@cli.command()
@click.option('--keywords', help='Comma-separated keywords to test filtering')
def test_keywords(keywords: str):
    """Test keyword filtering to see job counts before running expensive LLM matching"""
    db = next(get_db())
    
    if not keywords:
        click.echo("❌ Please provide keywords with --keywords")
        return
    
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
    
    click.echo(f"🔍 Testing keyword filter: {', '.join(keyword_list)}")
    click.echo("=" * 50)
    
    # Get total counts
    total_jobs = crud.JobOperations.count_all(db)
    filtered_count = crud.JobOperations.count_jobs_with_keywords(db, keyword_list)
    
    click.echo(f"📊 Results:")
    click.echo(f"   Total jobs in database: {total_jobs:,}")
    click.echo(f"   Jobs matching keywords: {filtered_count:,}")
    click.echo(f"   Filtering efficiency: {((total_jobs - filtered_count) / total_jobs * 100):.1f}% reduction")
    
    if filtered_count > 0:
        # Show sample jobs
        sample_jobs = crud.JobOperations.get_jobs_with_keywords(db, keyword_list, limit=5)
        click.echo(f"\n📋 Sample matching jobs:")
        for job in sample_jobs:
            click.echo(f"   • {job.company.name}: {job.title}")
            if job.skills_extracted:
                matching_skills = [s for s in job.skills_extracted if any(k.lower() in s.lower() for k in keyword_list)]
                if matching_skills:
                    click.echo(f"     Skills: {', '.join(matching_skills[:3])}")
    
    # Cost estimation
    if filtered_count > 0:
        estimated_llm_calls = filtered_count
        estimated_cost = estimated_llm_calls * 0.002  # Rough estimate: $0.002 per LLM call
        total_cost = total_jobs * 0.002
        
        click.echo(f"\n💰 Cost Estimation:")
        click.echo(f"   Without filtering: ~${total_cost:.2f} (LLM calls for {total_jobs:,} jobs)")
        click.echo(f"   With filtering: ~${estimated_cost:.2f} (LLM calls for {filtered_count:,} jobs)")
        click.echo(f"   Savings: ~${(total_cost - estimated_cost):.2f} ({((total_cost - estimated_cost) / total_cost * 100):.1f}%)")
    
    click.echo(f"\n💡 Recommendations:")
    if filtered_count > 2000:
        click.echo("   • Consider adding more specific keywords to reduce job count")
        click.echo("   • Or increase --max-jobs limit for broader matching")
    elif filtered_count < 50:
        click.echo("   • Try broader keywords or fewer filters")
        click.echo("   • Check if keywords match your job database content")
    else:
        click.echo("   • Good keyword balance for efficient matching!")
        click.echo(f"   • Run: python -m cli.admin match --resume-id=<ID> --keywords=\"{keywords}\"")

@cli.command()
def analyze_jobs():
    """Analyze job database to suggest good filtering keywords"""
    db = next(get_db())
    
    click.echo("📊 Analyzing job database for keyword suggestions...")
    
    # Get top skills
    top_skills = crud.JobOperations.get_top_skills(db, limit=20)
    
    # Get company distribution
    companies = crud.CompanyOperations.get_all_with_job_counts(db, limit=10)
    
    # Get location distribution  
    locations = crud.JobOperations.get_top_locations(db, limit=10)
    
    click.echo(f"\n🔧 Top Skills (good for filtering):")
    for i, skill in enumerate(top_skills, 1):
        count = crud.JobOperations.count_jobs_with_keywords(db, [skill])
        click.echo(f"   {i:2d}. {skill:<20} ({count:,} jobs)")
    
    click.echo(f"\n🏢 Top Companies:")
    for company in companies:
        click.echo(f"   • {company.name:<20} ({company.job_count:,} jobs)")
    
    click.echo(f"\n📍 Top Locations:")
    for location, count in locations:
        click.echo(f"   • {location:<20} ({count:,} jobs)")
    
    click.echo(f"\n💡 Suggested keyword combinations:")
    click.echo(f"   • For Python developers: \"python,django,flask,fastapi\"")
    click.echo(f"   • For Frontend: \"react,vue,angular,javascript,typescript\"")
    click.echo(f"   • For DevOps: \"aws,docker,kubernetes,terraform,cicd\"")
    click.echo(f"   • For Data Science: \"python,sql,machine learning,pytorch,pandas\"")
    
    total_jobs = crud.JobOperations.count_all(db)
    click.echo(f"\n📈 Database Stats:")
    click.echo(f"   Total jobs: {total_jobs:,}")
    click.echo(f"   Companies: {len(companies):,}")
    click.echo(f"   Unique skills: {len(top_skills):,}")

@cli.command()
def status():
    """Show system status and statistics"""
    db = next(get_db())
    
    # Count records
    company_count = db.query(crud.models.Company).count()
    job_count = db.query(crud.models.Job).count()
    resume_count = db.query(crud.models.Resume).count()
    match_count = db.query(crud.models.JobMatch).count()
    
    click.echo("📊 RoleRadar Status")
    click.echo("=" * 40)
    click.echo(f"Companies:     {company_count:,}")
    click.echo(f"Jobs:          {job_count:,}")
    click.echo(f"Resumes:       {resume_count:,}")
    click.echo(f"Matches:       {match_count:,}")
    
    # Recent activity
    recent_jobs = db.query(crud.models.Job).order_by(
        crud.models.Job.scraped_at.desc()
    ).limit(5).all()
    
    if recent_jobs:
        click.echo("\n🕐 Recent Jobs:")
        for job in recent_jobs:
            click.echo(f"   {job.company.name}: {job.title}")

if __name__ == '__main__':
    cli()
```

---

## 🚀 Phase 2: AWS Deployment (Like HPC!)

### Step 2.1: Create Your "Personal Compute Node"

Just like requesting a compute node on an HPC cluster, you're requesting a virtual machine on AWS:

```bash
# On HPC: sbatch --nodes=1 --mem=8GB job.slurm
# On AWS: Launch instance with 8GB RAM

# 1. Launch your "compute node" (EC2 instance)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-group-ids sg-your-security-group

# 2. Get your "node's IP address"
aws ec2 describe-instances --query 'Reservations[0].Instances[0].PublicIpAddress'
# Output: "3.144.23.156"
```

### Step 2.2: "SSH into Your Compute Node"

```bash
# Just like HPC: ssh username@compute-node-15.cluster.edu
ssh -i your-key.pem ubuntu@3.144.23.156

# You're now on your dedicated AWS "compute node"!
# Unlike HPC, this machine is 100% yours, 24/7
```

### Step 2.3: Set Up Your Environment (Better than HPC!)

```bash
# On HPC: module load python/3.9 (limited, shared environment)
# On AWS: Install whatever you want! (you're root)

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker (your application runtime)
sudo apt install -y docker.io docker-compose git

# Add yourself to docker group (no more sudo needed)
sudo usermod -aG docker ubuntu
# Log out and back in for changes to take effect

# Verify installation
docker --version
docker-compose --version
```

### Step 2.4: Deploy Your Application

```bash
# Clone your code (like copying to HPC scratch space)
git clone https://github.com/yourusername/roleradar.git
cd roleradar

# Deploy your entire stack (like running a job, but it stays up!)
docker-compose up -d

# Check status (like checking job queue)
docker-compose ps

# Your app is now live!
# Frontend: http://3.144.23.156:3000
# API: http://3.144.23.156:8000
# Database: Running internally
```

### Step 2.5: Application Management (Like Job Control)

```bash
# Check logs (like checking job output)
docker-compose logs -f

# Stop application (like canceling a job)
docker-compose down

# Restart application (like resubmitting a job)
docker-compose up -d

# Update your code (like modifying and rerunning)
git pull origin main
docker-compose restart

# Admin commands (your CLI tools)
docker-compose exec app python -m cli.admin status
docker-compose exec app python -m cli.admin scrape --company netflix
```

---

## 🎨 Phase 3: Frontend Development

### Step 3.1: Create Frontend Directory Structure

```bash
# In your main roleradar directory
mkdir frontend
cd frontend

# Create Next.js application (choose No for TypeScript, Yes for Tailwind, No for App Router)
npx create-next-app@latest . --typescript=false --tailwind=true --app=false

# Install additional dependencies
npm install @tanstack/react-query axios lucide-react recharts
```

**Your project structure should now look like:**
```
roleradar/
├── docker-compose.yml
├── database/
├── src/               # Your existing ML pipeline
├── cli/
├── api/
└── frontend/          # New frontend directory
    ├── package.json
    ├── pages/
    ├── components/
    ├── styles/
    └── public/
```

### Step 3.2: Frontend Configuration Files

#### Next.js Configuration
```javascript
// frontend/next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NODE_ENV === 'production' 
          ? 'http://api:8000/api/:path*'  // Docker internal network
          : 'http://localhost:8000/api/:path*'  // Local development
      },
    ]
  },
}

module.exports = nextConfig
```

#### Environment Variables
```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

#### Package.json Scripts
```json
{
  "name": "roleradar-frontend",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start -p 3000",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "^18",
    "react-dom": "^18",
    "@tanstack/react-query": "^5.0.0",
    "axios": "^1.6.0",
    "lucide-react": "^0.400.0",
    "recharts": "^2.8.0"
  },
  "devDependencies": {
    "autoprefixer": "^10",
    "postcss": "^8",
    "tailwindcss": "^3"
  }
}
```

### Step 3.3: Docker Configuration for Frontend

#### Frontend Dockerfile
```dockerfile
# frontend/Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
```

#### Docker Compose Integration
**Update your main `docker-compose.yml` to include frontend:**

```yaml
# docker-compose.yml (add this to your existing file)
services:
  # ... your existing db and api services ...

  # Frontend service
  frontend:
    build: 
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000/api
      - NODE_ENV=production
    depends_on:
      - api
    networks:
      - roleradar-network

  # Update your API service to use the network
  api:
    # ... existing config ...
    networks:
      - roleradar-network

  db:
    # ... existing config ...
    networks:
      - roleradar-network

# Add network configuration
networks:
  roleradar-network:
    driver: bridge
```

### Step 3.4: API Service Layer

```javascript
// frontend/services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

class RoleRadarAPI {
  // Resume operations
  async uploadResume(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/resumes/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async getResumes() {
    const response = await api.get('/resumes');
    return response.data;
  }

  async getResume(resumeId) {
    const response = await api.get(`/resumes/${resumeId}`);
    return response.data;
  }

  // Job operations
  async searchJobs(filters = {}) {
    const response = await api.get('/jobs', { params: filters });
    return response.data;
  }

  async getJob(jobId) {
    const response = await api.get(`/jobs/${jobId}`);
    return response.data;
  }

  async getCompanies() {
    const response = await api.get('/companies');
    return response.data;
  }

  // Matching operations
  async getMatches(resumeId, filters = {}) {
    const response = await api.get(`/matches`, { 
      params: { resume_id: resumeId, ...filters } 
    });
    return response.data;
  }

  async triggerMatching(resumeId, options = {}) {
    const response = await api.post('/matches/run', {
      resume_id: resumeId,
      ...options
    });
    return response.data;
  }

  // System status
  async getStatus() {
    const response = await api.get('/status');
    return response.data;
  }
}

export default new RoleRadarAPI();
```

### Step 3.5: Core React Components

#### Main Layout Component
```javascript
// frontend/components/Layout.js
import Link from 'next/link';
import { useState } from 'react';
import { Menu, X, Briefcase, Upload, Search, BarChart3 } from 'lucide-react';

export default function Layout({ children }) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navigation = [
    { name: 'Dashboard', href: '/', icon: BarChart3 },
    { name: 'Upload Resume', href: '/upload', icon: Upload },
    { name: 'Browse Jobs', href: '/jobs', icon: Search },
    { name: 'My Matches', href: '/matches', icon: Briefcase },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <Link href="/" className="flex items-center">
                <Briefcase className="h-8 w-8 text-blue-600" />
                <span className="ml-2 text-xl font-bold text-gray-900">RoleRadar</span>
              </Link>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-8">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className="text-gray-500 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium flex items-center"
                >
                  <item.icon className="h-4 w-4 mr-2" />
                  {item.name}
                </Link>
              ))}
            </div>

            {/* Mobile menu button */}
            <div className="md:hidden flex items-center">
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="text-gray-400 hover:text-gray-500"
              >
                {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className="text-gray-500 hover:text-gray-900 block px-3 py-2 rounded-md text-base font-medium"
                  onClick={() => setIsMenuOpen(false)}
                >
                  <item.icon className="h-4 w-4 inline mr-2" />
                  {item.name}
                </Link>
              ))}
            </div>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl py-6 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}
```

#### Resume Upload Component
```javascript
// frontend/components/ResumeUpload.js
import { useState, useCallback } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import api from '../services/api';

export default function ResumeUpload({ onUploadSuccess }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      await uploadFile(files[0]);
    }
  }, []);

  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    if (file) {
      await uploadFile(file);
    }
  };

  const uploadFile = async (file) => {
    // Validate file type
    const allowedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!allowedTypes.includes(file.type)) {
      setUploadStatus({ type: 'error', message: 'Please upload a PDF or Word document' });
      return;
    }

    setIsUploading(true);
    setUploadStatus(null);

    try {
      const result = await api.uploadResume(file);
      setUploadStatus({ 
        type: 'success', 
        message: `Resume uploaded successfully! Parsed ${result.skills_count || 0} skills.` 
      });
      
      if (onUploadSuccess) {
        onUploadSuccess(result);
      }
    } catch (error) {
      setUploadStatus({ 
        type: 'error', 
        message: `Upload failed: ${error.response?.data?.detail || error.message}` 
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-lg mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-lg p-6 transition-colors ${
          isDragOver 
            ? 'border-blue-400 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
      >
        <div className="text-center">
          {isUploading ? (
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          ) : (
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
          )}
          
          <div className="mt-4">
            <label htmlFor="file-upload" className="cursor-pointer">
              <span className="mt-2 block text-sm font-medium text-gray-900">
                {isUploading ? 'Processing resume...' : 'Drop your resume here, or click to browse'}
              </span>
              <span className="mt-1 block text-xs text-gray-500">
                PDF, DOC, or DOCX up to 10MB
              </span>
            </label>
            <input
              id="file-upload"
              name="file-upload"
              type="file"
              className="sr-only"
              accept=".pdf,.doc,.docx"
              onChange={handleFileSelect}
              disabled={isUploading}
            />
          </div>
        </div>
      </div>

      {/* Status Messages */}
      {uploadStatus && (
        <div className={`mt-4 p-4 rounded-md ${
          uploadStatus.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
        }`}>
          <div className="flex">
            {uploadStatus.type === 'success' ? (
              <CheckCircle className="h-5 w-5 mr-2" />
            ) : (
              <AlertCircle className="h-5 w-5 mr-2" />
            )}
            <span className="text-sm">{uploadStatus.message}</span>
          </div>
        </div>
      )}
    </div>
  );
}
```

#### Job Keyword Filter Component
```javascript
// frontend/components/JobKeywordFilter.js
import { useState, useEffect } from 'react';
import { Search, Tag, TrendingUp } from 'lucide-react';
import api from '../services/api';

export default function JobKeywordFilter({ onKeywordsSelected, totalJobs }) {
  const [searchInput, setSearchInput] = useState('');
  const [selectedKeywords, setSelectedKeywords] = useState([]);
  const [suggestedKeywords, setSuggestedKeywords] = useState([]);
  const [filteredJobCount, setFilteredJobCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  // Popular tech keywords - could be loaded from API
  const popularKeywords = [
    'Python', 'JavaScript', 'React', 'Node.js', 'AWS', 'Docker', 'Kubernetes',
    'Machine Learning', 'Data Science', 'SQL', 'PostgreSQL', 'MongoDB',
    'Frontend', 'Backend', 'Full Stack', 'DevOps', 'Cloud', 'API',
    'TypeScript', 'Vue.js', 'Angular', 'Express', 'Flask', 'Django'
  ];

  useEffect(() => {
    loadSuggestedKeywords();
  }, []);

  useEffect(() => {
    if (selectedKeywords.length > 0) {
      updateJobCount();
    } else {
      setFilteredJobCount(0);
    }
  }, [selectedKeywords]);

  const loadSuggestedKeywords = async () => {
    try {
      // Get top skills from current job database
      const topSkills = await api.getTopJobSkills(20);
      setSuggestedKeywords(topSkills);
    } catch (error) {
      // Fallback to popular keywords
      setSuggestedKeywords(popularKeywords.slice(0, 12));
    }
  };

  const updateJobCount = async () => {
    setIsLoading(true);
    try {
      const count = await api.getFilteredJobCount(selectedKeywords);
      setFilteredJobCount(count);
      onKeywordsSelected(selectedKeywords, count);
    } catch (error) {
      console.error('Failed to get job count:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const addKeyword = (keyword) => {
    if (!selectedKeywords.includes(keyword) && keyword.trim()) {
      setSelectedKeywords([...selectedKeywords, keyword.trim()]);
    }
    setSearchInput('');
  };

  const removeKeyword = (keyword) => {
    setSelectedKeywords(selectedKeywords.filter(k => k !== keyword));
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchInput.trim()) {
      addKeyword(searchInput);
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Input */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Add Keywords to Filter Jobs
        </label>
        <form onSubmit={handleSearchSubmit} className="flex space-x-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Type skills, technologies, job titles..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Add
          </button>
        </form>
      </div>

      {/* Selected Keywords */}
      {selectedKeywords.length > 0 && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Selected Keywords ({selectedKeywords.length})
          </label>
          <div className="flex flex-wrap gap-2">
            {selectedKeywords.map((keyword) => (
              <span
                key={keyword}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-800"
              >
                <Tag className="w-3 h-3 mr-1" />
                {keyword}
                <button
                  onClick={() => removeKeyword(keyword)}
                  className="ml-2 text-blue-600 hover:text-blue-800"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
          
          {/* Job Count Display */}
          <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center">
              <TrendingUp className="w-4 h-4 text-green-600 mr-2" />
              <span className="text-green-800 font-medium">
                {isLoading ? (
                  'Counting jobs...'
                ) : (
                  <>
                    {filteredJobCount:,} jobs match your keywords 
                    <span className="text-green-600 text-sm ml-1">
                      ({Math.round((filteredJobCount / totalJobs) * 100)}% of total)
                    </span>
                  </>
                )}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Suggested Keywords */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Popular Keywords (click to add)
        </label>
        <div className="flex flex-wrap gap-2">
          {suggestedKeywords
            .filter(keyword => !selectedKeywords.includes(keyword))
            .map((keyword) => (
              <button
                key={keyword}
                onClick={() => addKeyword(keyword)}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
              >
                {keyword}
              </button>
            ))}
        </div>
      </div>

      {/* Help Text */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-blue-800 font-medium mb-2">💡 Pro Tips:</h4>
        <ul className="text-blue-700 text-sm space-y-1">
          <li>• Use specific skills (Python, React) rather than generic terms (programming)</li>
          <li>• Include both technologies and job types you're interested in</li>
          <li>• More keywords = more targeted matches but fewer total results</li>
          <li>• Aim for 3-8 keywords for optimal balance</li>
        </ul>
      </div>
    </div>
  );
}
```

#### Job Results Table Component
```javascript
// frontend/components/JobResultsTable.js
import { useState } from 'react';
import { ExternalLink, ChevronUp, ChevronDown, DollarSign, MapPin, Calendar } from 'lucide-react';

export default function JobResultsTable({ results }) {
  const [sortBy, setSortBy] = useState('match_score');
  const [sortDirection, setSortDirection] = useState('desc');

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortDirection('desc');
    }
  };

  const sortedResults = [...results].sort((a, b) => {
    let aVal = a[sortBy];
    let bVal = b[sortBy];
    
    // Handle salary sorting (use midpoint of range)
    if (sortBy === 'salary') {
      aVal = a.salary_max ? (a.salary_min + a.salary_max) / 2 : a.salary_min || 0;
      bVal = b.salary_max ? (b.salary_min + b.salary_max) / 2 : b.salary_min || 0;
    }
    
    // Handle string comparison
    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = bVal.toLowerCase();
    }
    
    if (sortDirection === 'asc') {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });

  const getMatchScoreColor = (score) => {
    if (score >= 80) return 'text-green-700 bg-green-100';
    if (score >= 65) return 'text-blue-700 bg-blue-100';
    if (score >= 50) return 'text-yellow-700 bg-yellow-100';
    return 'text-red-700 bg-red-100';
  };

  const getMatchScoreLabel = (score) => {
    if (score >= 70) return 'Auto Apply';
    if (score >= 50) return 'Good Match';
    return 'Consider';
  };

  const formatSalary = (job) => {
    if (job.salary_max && job.salary_min) {
      return `${(job.salary_min / 1000).toFixed(0)}k - ${(job.salary_max / 1000).toFixed(0)}k`;
    } else if (job.salary_min) {
      return `${(job.salary_min / 1000).toFixed(0)}k+`;
    }
    return 'Not specified';
  };

  const SortHeader = ({ column, children }) => (
    <th
      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
      onClick={() => handleSort(column)}
    >
      <div className="flex items-center space-x-1">
        <span>{children}</span>
        {sortBy === column && (
          sortDirection === 'asc' ? 
            <ChevronUp className="w-4 h-4" /> : 
            <ChevronDown className="w-4 h-4" />
        )}
      </div>
    </th>
  );

  return (
    <div className="space-y-4">
      {/* Results Summary */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          Found {results.length} Matching Jobs
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Auto Apply (70%+):</span>
            <span className="ml-2 font-medium text-green-600">
              {results.filter(r => r.match_score >= 70).length}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Good Match (50%+):</span>
            <span className="ml-2 font-medium text-blue-600">
              {results.filter(r => r.match_score >= 50 && r.match_score < 70).length}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Avg Match Score:</span>
            <span className="ml-2 font-medium">
              {Math.round(results.reduce((sum, r) => sum + r.match_score, 0) / results.length)}%
            </span>
          </div>
          <div>
            <span className="text-gray-500">With Salary Info:</span>
            <span className="ml-2 font-medium">
              {results.filter(r => r.salary_min || r.salary_max).length}
            </span>
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <SortHeader column="match_score">Match %</SortHeader>
              <SortHeader column="company_name">Company</SortHeader>
              <SortHeader column="title">Job Title</SortHeader>
              <SortHeader column="salary">Salary</SortHeader>
              <SortHeader column="location">Location</SortHeader>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedResults.map((job, index) => (
              <tr key={job.id || index} className="hover:bg-gray-50">
                {/* Match Score */}
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getMatchScoreColor(job.match_score)}`}>
                      {job.match_score}%
                    </span>
                    <span className="ml-2 text-xs text-gray-500">
                      {getMatchScoreLabel(job.match_score)}
                    </span>
                  </div>
                </td>

                {/* Company */}
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">
                    {job.company_name}
                  </div>
                </td>

                {/* Job Title */}
                <td className="px-6 py-4">
                  <div className="text-sm font-medium text-gray-900">
                    {job.title}
                  </div>
                  {job.employment_type && (
                    <div className="text-sm text-gray-500">
                      {job.employment_type}
                    </div>
                  )}
                </td>

                {/* Salary */}
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center text-sm text-gray-900">
                    <DollarSign className="w-4 h-4 mr-1 text-gray-400" />
                    {formatSalary(job)}
                  </div>
                </td>

                {/* Location */}
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center text-sm text-gray-900">
                    <MapPin className="w-4 h-4 mr-1 text-gray-400" />
                    {job.location || 'Remote/Unspecified'}
                  </div>
                </td>

                {/* Action */}
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <a
                    href={job.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                  >
                    Apply
                    <ExternalLink className="ml-2 w-4 h-4" />
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Export Options */}
      <div className="flex justify-between items-center pt-4">
        <div className="text-sm text-gray-500">
          Showing {results.length} results
        </div>
        <div className="space-x-2">
          <button
            onClick={() => {
              const csv = convertToCSV(sortedResults);
              downloadCSV(csv, 'job-matches.csv');
            }}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm text-gray-700 bg-white hover:bg-gray-50"
          >
            Export CSV
          </button>
          <button
            onClick={() => {
              const highMatches = sortedResults.filter(r => r.match_score >= 70);
              window.open('/bulk-apply?jobs=' + highMatches.map(j => j.id).join(','), '_blank');
            }}
            className="px-3 py-2 border border-transparent rounded-md text-sm text-white bg-green-600 hover:bg-green-700"
            disabled={sortedResults.filter(r => r.match_score >= 70).length === 0}
          >
            Bulk Apply to Top Matches ({sortedResults.filter(r => r.match_score >= 70).length})
          </button>
        </div>
      </div>
    </div>
  );
}

// Helper functions
function convertToCSV(data) {
  const headers = ['Company', 'Job Title', 'Match Score', 'Salary', 'Location', 'URL'];
  const rows = data.map(job => [
    job.company_name,
    job.title,
    `${job.match_score}%`,
    formatSalary(job),
    job.location || 'Remote/Unspecified',
    job.url
  ]);
  
  const csvContent = [headers, ...rows]
    .map(row => row.map(field => `"${field}"`).join(','))
    .join('\n');
  
  return csvContent;
}

function formatSalary(job) {
  if (job.salary_max && job.salary_min) {
    return `${(job.salary_min / 1000).toFixed(0)}k - ${(job.salary_max / 1000).toFixed(0)}k`;
  } else if (job.salary_min) {
    return `${(job.salary_min / 1000).toFixed(0)}k+`;
  }
  return 'Not specified';
}

function downloadCSV(csv, filename) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  if (link.download !== undefined) {
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}
```

### Step 3.6: Main Pages

#### Homepage/Dashboard
```javascript
// frontend/pages/index.js
import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { BarChart3, Briefcase, Upload, Search } from 'lucide-react';
import Link from 'next/link';
import api from '../services/api';

export default function Dashboard() {
  const [stats, setStats] = useState({
    resumes: 0,
    jobs: 0,
    matches: 0,
    companies: 0
  });

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const status = await api.getStatus();
      setStats(status);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const quickActions = [
    {
      name: 'Upload Resume',
      description: 'Upload and parse a new resume',
      href: '/upload',
      icon: Upload,
      color: 'bg-blue-500 hover:bg-blue-600',
    },
    {
      name: 'Browse Jobs',
      description: 'Search through available positions',
      href: '/jobs',
      icon: Search,
      color: 'bg-green-500 hover:bg-green-600',
    },
    {
      name: 'View Matches',
      description: 'See your job recommendations',
      href: '/matches',
      icon: Briefcase,
      color: 'bg-purple-500 hover:bg-purple-600',
    },
  ];

  return (
    <Layout>
      <div className="px-4 sm:px-0">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Welcome to RoleRadar - your AI-powered job matching platform
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {[
            { name: 'Resumes', value: stats.resumes, icon: Upload },
            { name: 'Jobs Available', value: stats.jobs, icon: Briefcase },
            { name: 'Total Matches', value: stats.matches, icon: BarChart3 },
            { name: 'Companies', value: stats.companies, icon: Search },
          ].map((stat) => (
            <div key={stat.name} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <stat.icon className="h-8 w-8 text-gray-400" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">{stat.name}</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {stat.value?.toLocaleString() || 0}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {quickActions.map((action) => (
              <Link key={action.name} href={action.href}>
                <div className={`${action.color} rounded-lg p-6 text-white cursor-pointer transition-colors`}>
                  <action.icon className="h-8 w-8 mb-3" />
                  <h3 className="text-lg font-semibold mb-2">{action.name}</h3>
                  <p className="text-sm opacity-90">{action.description}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
}
```

#### Main Job Matching Page (Upload + Filter + Results)
```javascript
// frontend/pages/index.js - Replace the dashboard with the main workflow
import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import ResumeUpload from '../components/ResumeUpload';
import JobKeywordFilter from '../components/JobKeywordFilter';
import JobResultsTable from '../components/JobResultsTable';
import { Play, FileText, Filter, Table } from 'lucide-react';
import api from '../services/api';

export default function JobMatcher() {
  const [currentStep, setCurrentStep] = useState(1);
  const [uploadedResume, setUploadedResume] = useState(null);
  const [selectedKeywords, setSelectedKeywords] = useState([]);
  const [isMatching, setIsMatching] = useState(false);
  const [matchResults, setMatchResults] = useState([]);
  const [stats, setStats] = useState({ totalJobs: 0, filteredJobs: 0 });

  const steps = [
    { id: 1, name: 'Upload Resume', icon: FileText, status: uploadedResume ? 'complete' : 'current' },
    { id: 2, name: 'Filter Jobs', icon: Filter, status: uploadedResume ? (selectedKeywords.length > 0 ? 'complete' : 'current') : 'upcoming' },
    { id: 3, name: 'View Results', icon: Table, status: matchResults.length > 0 ? 'complete' : 'upcoming' },
  ];

  useEffect(() => {
    loadJobStats();
  }, []);

  const loadJobStats = async () => {
    try {
      const status = await api.getStatus();
      setStats({ totalJobs: status.jobs || 0, filteredJobs: 0 });
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const handleUploadSuccess = (resume) => {
    setUploadedResume(resume);
    setCurrentStep(2);
  };

  const handleKeywordsSelected = (keywords, filteredCount) => {
    setSelectedKeywords(keywords);
    setStats(prev => ({ ...prev, filteredJobs: filteredCount }));
    if (keywords.length > 0) {
      setCurrentStep(3);
    }
  };

  const runMatching = async () => {
    if (!uploadedResume || selectedKeywords.length === 0) return;

    setIsMatching(true);
    try {
      const results = await api.runFilteredMatching({
        resume_id: uploadedResume.id,
        keywords: selectedKeywords,
        min_score: 50
      });
      setMatchResults(results);
    } catch (error) {
      console.error('Matching failed:', error);
      alert('Matching failed. Please try again.');
    } finally {
      setIsMatching(false);
    }
  };

  const getStepStatus = (step) => {
    if (step.status === 'complete') return 'bg-green-500 text-white';
    if (step.status === 'current') return 'bg-blue-500 text-white';
    return 'bg-gray-300 text-gray-500';
  };

  return (
    <Layout>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">AI Job Matcher</h1>
          <p className="mt-2 text-gray-600">
            Upload your resume, filter relevant jobs, and get instant AI-powered matches
          </p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <nav aria-label="Progress">
            <ol className="flex items-center justify-center space-x-8">
              {steps.map((step, stepIdx) => (
                <li key={step.name} className="flex items-center">
                  <div className={`flex items-center justify-center w-10 h-10 rounded-full ${getStepStatus(step)}`}>
                    <step.icon className="w-6 h-6" />
                  </div>
                  <span className="ml-2 text-sm font-medium text-gray-500">{step.name}</span>
                  {stepIdx !== steps.length - 1 && (
                    <div className="ml-8 w-8 h-px bg-gray-300" />
                  )}
                </li>
              ))}
            </ol>
          </nav>
        </div>

        {/* Step 1: Resume Upload */}
        <div className="mb-8">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <FileText className="w-5 h-5 mr-2" />
              Step 1: Upload Your Resume
            </h2>
            
            {!uploadedResume ? (
              <ResumeUpload onUploadSuccess={handleUploadSuccess} />
            ) : (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center">
                  <div className="bg-green-500 rounded-full p-1 mr-3">
                    <FileText className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <p className="text-green-800 font-medium">{uploadedResume.filename}</p>
                    <p className="text-green-600 text-sm">
                      Parsed {uploadedResume.skills_count || 0} skills, {uploadedResume.experience_years || 0} years experience
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Step 2: Keyword Filtering */}
        {uploadedResume && (
          <div className="mb-8">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Filter className="w-5 h-5 mr-2" />
                Step 2: Filter Jobs by Keywords
                <span className="ml-2 text-sm text-gray-500">
                  ({stats.totalJobs:,} total jobs available)
                </span>
              </h2>
              
              <JobKeywordFilter 
                onKeywordsSelected={handleKeywordsSelected}
                totalJobs={stats.totalJobs}
              />
            </div>
          </div>
        )}

        {/* Step 3: Run Matching & Results */}
        {uploadedResume && selectedKeywords.length > 0 && (
          <div className="mb-8">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold flex items-center">
                  <Table className="w-5 h-5 mr-2" />
                  Step 3: Job Matches
                  {stats.filteredJobs > 0 && (
                    <span className="ml-2 text-sm text-gray-500">
                      (Matching against {stats.filteredJobs:,} filtered jobs)
                    </span>
                  )}
                </h2>
                
                {matchResults.length === 0 && (
                  <button
                    onClick={runMatching}
                    disabled={isMatching}
                    className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-6 py-2 rounded-lg flex items-center"
                  >
                    {isMatching ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Running AI Analysis...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Run AI Matching
                      </>
                    )}
                  </button>
                )}
              </div>

              {matchResults.length > 0 && (
                <JobResultsTable results={matchResults} />
              )}
            </div>
          </div>
        )}

        {/* Performance Info */}
        {selectedKeywords.length > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-8">
            <h3 className="text-blue-800 font-medium mb-2">⚡ Speed Optimization</h3>
            <p className="text-blue-700 text-sm">
              By filtering with keywords, you're analyzing {stats.filteredJobs:,} jobs instead of {stats.totalJobs:,} jobs.
              This saves ~{Math.round((1 - stats.filteredJobs / stats.totalJobs) * 100)}% processing time and LLM API costs!
            </p>
          </div>
        )}
      </div>
    </Layout>
  );
}
```

### Step 3.7: Development vs Production Setup

#### Development Mode (Local)
```bash
# Terminal 1: Start backend services
docker-compose up db api

# Terminal 2: Start frontend in development mode
cd frontend
npm run dev

# Frontend runs at: http://localhost:3000
# API runs at: http://localhost:8000
# Database runs at: localhost:5432
```

#### Production Mode (Docker)
```bash
# Build and start everything together
docker-compose up --build

# All services run in containers:
# Frontend: http://localhost:3000
# API: http://localhost:8000 (internal)
# Database: internal only
```

### Step 3.8: Frontend Testing Commands

```bash
# Inside frontend directory

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Check for code issues
npm run lint

# Install new packages
npm install package-name

# Test API connection
curl http://localhost:3000/api/status
```

### Step 3.9: Docker Networking Explained

When everything runs in Docker:

```
Docker Internal Network (roleradar-network):
├── frontend container (port 3000)
├── api container (port 8000) 
└── db container (port 5432)

External Access:
├── Users visit: http://your-server:3000 (frontend)
├── Frontend calls: http://api:8000 (internal Docker network)
└── API calls: http://db:5432 (internal Docker network)
```

**Key Point**: Frontend calls API using Docker service name (`api`) not localhost when both are in containers!

---

## 🐳 Docker Configuration

### docker-compose.yml

```yaml
version: '3.8'

services:
  # Database
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: roleradar
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD:-secretpassword}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d roleradar"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Backend API
  api:
    build: 
      context: .
      dockerfile: Dockerfile.api
    environment:
      - DATABASE_URL=postgresql://admin:${DB_PASSWORD:-secretpassword}@db:5432/roleradar
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./src:/app/src
      - ./cli:/app/cli
      - ./uploads:/app/uploads
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000/api
    depends_on:
      - api

  # Background worker (for async jobs)
  worker:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - DATABASE_URL=postgresql://admin:${DB_PASSWORD:-secretpassword}@db:5432/roleradar
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./src:/app/src
      - ./cli:/app/cli
    depends_on:
      db:
        condition: service_healthy
    command: python -m cli.admin worker  # Background processing

volumes:
  postgres_data:
```

---

## 🚀 Quick Start Guide

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/yourusername/roleradar.git
cd roleradar

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start everything
docker-compose up -d

# 4. Initialize database
docker-compose exec api python -m cli.admin add-company \
    --name "Netflix" \
    --sitemap-url "https://explore.jobs.netflix.net/careers/sitemap.xml"

# 5. Test scraping
docker-compose exec api python -m cli.admin scrape --company Netflix --max-jobs 10

# 6. Upload test resume
docker-compose exec api python -m cli.admin parse-resume test_resume.pdf

# 7. Run matching
docker-compose exec api python -m cli.admin match --resume-id <resume-id> --company Netflix

# 8. Check results
open http://localhost:3000
```

### AWS Deployment

```bash
# 1. Launch EC2 instance (your "compute node")
aws ec2 run-instances --image-id ami-0c02fb55956c7d316 --instance-type t3.medium

# 2. SSH into your instance (like HPC login)
ssh -i your-key.pem ubuntu@YOUR-EC2-IP

# 3. Install Docker (like loading modules on HPC)
sudo apt update && sudo apt install -y docker.io docker-compose git
sudo usermod -aG docker ubuntu
# Logout and login again

# 4. Deploy your application (like running a job)
git clone https://github.com/yourusername/roleradar.git
cd roleradar
docker-compose -f docker-compose.prod.yml up -d

# 5. Your app is live!
# Visit: http://YOUR-EC2-IP:3000
```

---

## 📊 Success Metrics

### Technical KPIs
- **Resume Processing**: <30 seconds per resume
- **Job Scraping**: >95% success rate
- **Matching Speed**: <10 seconds for 1000+ jobs
- **System Uptime**: 99.9% availability

### Business KPIs
- **Match Accuracy**: User feedback on recommendations
- **Application Success**: Interview rate for auto-apply jobs
- **User Engagement**: Resume iterations and improvements

---

## 🛠️ Development Workflow

### Daily Development
```bash
# Make changes to code
git add . && git commit -m "Add feature X"

# Test locally
docker-compose restart api

# Deploy to AWS (like resubmitting HPC job)
ssh ubuntu@YOUR-EC2-IP 'cd roleradar && git pull && docker-compose restart'
```

### Adding New Features
1. **Backend**: Add API endpoints in `api/routes/`
2. **Database**: Create migration in `database/migrations/`
3. **CLI**: Add commands in `cli/admin.py`
4. **Frontend**: Add components in `frontend/components/`

---

## 🎓 Learning Outcomes

This project teaches you:

- **Database Design**: PostgreSQL, SQLAlchemy, migrations
- **API Development**: FastAPI, REST principles, documentation
- **Frontend Development**: React, state management, API integration
- **DevOps**: Docker, container orchestration, cloud deployment
- **ML Engineering**: Model integration, batch processing, pipeline design
- **System Architecture**: Microservices, separation of concerns
- **Cloud Computing**: AWS services, infrastructure management

You're building a production-grade application using industry-standard tools and practices! 🚀

---

## 📞 Support

### Troubleshooting
- **Database Issues**: Check `docker-compose logs db`
- **API Errors**: Check `docker-compose logs api`
- **Frontend Issues**: Check `docker-compose logs frontend`

### Common Commands
```bash
# View all logs
docker-compose logs -f

# Reset database
docker-compose down -v && docker-compose up -d

# Run CLI commands
docker-compose exec api python -m cli.admin [command]

# Access database directly
docker-compose exec db psql -U admin -d roleradar
```

Ready to transform your scraper into a full-stack application! 🎯