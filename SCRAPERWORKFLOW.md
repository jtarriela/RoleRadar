# Career Site Scraper - High-Level Workflow

## Project Overview
A automated system that scrapes job postings from company career sites, parses user resumes, and intelligently matches candidates to relevant positions with actionable recommendations.

## System Architecture

### Core Components
1. **Resume Processing Engine** - LLM-powered resume parsing and standardization
2. **Web Scraping Pipeline** - Automated job posting extraction and processing
3. **Matching Engine** - AI-driven job-resume compatibility scoring
4. **Recommendation System** - Resume optimization suggestions
5. **Data Storage Layer** - Structured data persistence
6. **Output Generator** - CSV reports and analytics

## Detailed Workflow

### Phase 1: Resume Processing Pipeline

#### 1.1 Resume Ingestion
```
Input: PDF/DOCX/TXT resume files
├── File validation and format detection
├── Text extraction (OCR if needed)
└── Raw text normalization
```

#### 1.2 LLM Resume Parsing
```
Raw Resume Text → LLM Parser → Structured JSON
```

**Target JSON Schema:**
```json
{
  "personal_info": {
    "name": "",
    "email": "",
    "phone": "",
    "location": ""
  },
  "experience": [
    {
      "title": "",
      "company": "",
      "duration": "",
      "responsibilities": [],
      "technologies": []
    }
  ],
  "education": [
    {
      "degree": "",
      "institution": "",
      "graduation_date": "",
      "gpa": ""
    }
  ],
  "skills": {
    "technical": [],
    "soft": [],
    "certifications": []
  },
  "projects": [],
  "keywords": []
}
```

#### 1.3 Resume Storage
- Store parsed resumes with unique identifiers
- Support multiple resume versions per user
- Maintain parsing metadata and timestamps

### Phase 2: Job Site Scraping Pipeline

#### 2.1 Sitemap Discovery
```
Company URLs → sitemap-extract → Job Listing URLs
```

**Process:**
1. Extract XML sitemaps from target company sites
2. Filter URLs matching job posting patterns
3. Deduplicate and validate URLs
4. Store in scraping queue with priority scoring

#### 2.2 Content Extraction
```
Job URLs → crawl4ai → Raw Markdown → Cleaned Content
```

**Markdown Cleaning Process:**
1. Remove all hyperlinked text
2. Strip URLs and reference links
3. Remove navigation elements and boilerplate
4. Preserve core job description content
5. Validate content quality (minimum word count, etc.)

#### 2.3 LLM Job Parsing
```
Cleaned Markdown → LLM Parser → Structured Job JSON
```

**Target Job JSON Schema:**
```json
{
  "job_info": {
    "title": "",
    "company": "",
    "location": "",
    "employment_type": "",
    "salary_range": "",
    "job_url": "",
    "posted_date": ""
  },
  "requirements": {
    "experience_level": "",
    "education": [],
    "required_skills": [],
    "preferred_skills": [],
    "certifications": []
  },
  "description": {
    "summary": "",
    "responsibilities": [],
    "benefits": [],
    "keywords": []
  },
  "parsing_metadata": {
    "confidence_score": 0.95,
    "timestamp": "",
    "source_quality": "high"
  }
}
```

### Phase 3: Semantic Matching Engine (JSON → Text → Embedding)

#### 3.1 Structured Text Conversion
```python
def format_resume_for_embedding(resume_json):
    sections = []
    
    # Professional experience
    if resume_json.get('experience'):
        exp_text = []
        for exp in resume_json['experience']:
            exp_text.append(f"{exp['title']} at {exp['company']}: {' '.join(exp['responsibilities'])}")
        sections.append(f"Professional Experience: {' '.join(exp_text)}")
    
    # Skills
    if resume_json.get('skills'):
        all_skills = resume_json['skills'].get('technical', []) + resume_json['skills'].get('soft', [])
        sections.append(f"Skills: {', '.join(all_skills)}")
    
    # Education
    if resume_json.get('education'):
        edu_text = [f"{edu['degree']} from {edu['institution']}" for edu in resume_json['education']]
        sections.append(f"Education: {', '.join(edu_text)}")
    
    # Projects
    if resume_json.get('projects'):
        proj_text = [f"{proj['name']}: {proj['description']}" for proj in resume_json['projects']]
        sections.append(f"Projects: {' '.join(proj_text)}")
    
    return "\n\n".join(sections)

def format_job_for_embedding(job_json):
    sections = []
    
    # Job title and summary
    if job_json.get('job_info'):
        sections.append(f"Position: {job_json['job_info']['title']}")
    
    if job_json.get('description', {}).get('summary'):
        sections.append(f"Role Summary: {job_json['description']['summary']}")
    
    # Requirements
    if job_json.get('requirements'):
        req = job_json['requirements']
        if req.get('required_skills'):
            sections.append(f"Required Skills: {', '.join(req['required_skills'])}")
        if req.get('preferred_skills'):
            sections.append(f"Preferred Skills: {', '.join(req['preferred_skills'])}")
        if req.get('experience_level'):
            sections.append(f"Experience Level: {req['experience_level']}")
    
    # Responsibilities
    if job_json.get('description', {}).get('responsibilities'):
        sections.append(f"Responsibilities: {' '.join(job_json['description']['responsibilities'])}")
    
    return "\n\n".join(sections)
```

#### 3.2 Embedding and Matching
```python
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = OpenAIEmbeddings()

def get_text_embedding(text):
    return embeddings.embed_query(text)

def calculate_match_score(resume_json, job_json):
    # Convert JSON to clean, structured text
    resume_text = format_resume_for_embedding(resume_json)
    job_text = format_job_for_embedding(job_json)
    
    # Generate embeddings
    resume_embedding = get_text_embedding(resume_text)
    job_embedding = get_text_embedding(job_text)
    
    # Calculate cosine similarity
    match_score = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    return round(match_score * 100, 2)  # Convert to percentage
```

#### 3.3 Why This Approach Works
- **Clean Input**: Removes contact info, navigation, and formatting noise
- **Structured Focus**: Only includes matching-relevant content
- **Scale Independence**: Cosine similarity measures semantic alignment, not document length
- **Semantic Capture**: Pre-trained embeddings understand domain relationships
- **Debuggable**: Can inspect exact text being compared for troubleshooting

#### 3.3 Business Logic Implementation
- **≥70% Match**: Auto-flag for application
- **65-69% Match**: Flag with confidence indicator
- **50-64% Match**: Generate improvement suggestions
- **<50% Match**: Archive with minimal processing

### Phase 4: Resume Optimization Engine

#### 4.1 Gap Analysis
For 50-64% matches, identify specific improvement areas:
```python
improvement_areas = {
    'missing_skills': required_skills - resume_skills,
    'weak_keywords': high_value_keywords - resume_keywords,
    'experience_gaps': analyze_experience_mismatch(),
    'education_recommendations': suggest_certifications()
}
```

#### 4.2 LLM-Powered Analysis
```python
def analyze_resume_with_llm(resume_text, jd_text, match_score):
    llm_context = """
    You are an experienced headhunter specializing in helping early-career professionals
    optimize their resumes for job applications. Your goal is to analyze a candidate's resume
    and compare it to a job description to provide insightful recommendations.
    
    Your output should be structured as follows:
    📅 Match Score: X%
    💪 Key Strengths:
    - Bullet point list of strengths based on resume-JD alignment
    🔍 Missing Skills:
    - Bullet point list of missing skills/keywords from JD
    💡 Recommendations:
    - Specific, actionable resume improvements
    💼 Sample Cover Letter:
    - Tailored cover letter highlighting relevant experience
    """
    
    prompt = f"{llm_context}\n\nMatch Score: {match_score}%\n\nResume:\n{resume_text}\n\nJob Description:\n{jd_text}\n\n"
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
```

### Phase 5: Data Pipeline & Storage

#### 5.1 Database Schema
```sql
-- Core Tables
resumes (id, user_id, version, parsed_data, created_at)
jobs (id, company, title, url, parsed_data, scraped_at)
matches (resume_id, job_id, score, status, recommendations)
companies (id, name, careers_url, last_scraped, scrape_status)

-- Analytics Tables
match_history (resume_id, job_id, score, timestamp)
improvement_tracking (resume_id, suggestions, implemented, score_change)
```

#### 5.2 Data Quality Assurance
- **Resume Validation**: Ensure all critical fields are parsed
- **Job Validation**: Verify job posting authenticity and completeness
- **Duplicate Detection**: Handle duplicate job postings across sites
- **Data Freshness**: Track and manage stale job postings

### Phase 6: Output Generation

#### 6.1 CSV Report Structure
```csv
resume_id,resume_version,company,job_title,match_score,status,salary_range,location,job_url,recommendations,last_updated
resume_a,v1,Google,Software Engineer,78,auto_apply,120k-150k,Mountain View CA,https://...,None,2024-01-15
resume_a,v1,Meta,Frontend Developer,62,improve_resume,110k-140k,Menlo Park CA,https://...,"Add React Native experience",2024-01-15
```

#### 6.2 Analytics Dashboard (Future Enhancement)
- Match score distributions
- Industry trending requirements
- Resume optimization impact tracking
- Application success rates

## Implementation Phases

### Phase 1: MVP (Weeks 1-2)
- Basic resume parsing with simple LLM prompts
- Single-site scraping proof-of-concept
- Simple keyword-based matching
- CSV output generation

### Phase 2: Core Features (Weeks 3-4)
- Multi-site scraping with sitemap-extract
- Enhanced LLM parsing with structured outputs
- Weighted scoring algorithm implementation
- Basic recommendation engine

### Phase 3: Production Ready (Weeks 5-6)
- Error handling and retry mechanisms
- Rate limiting and respectful scraping
- Data quality validation
- Performance optimization

### Phase 4: Advanced Features (Weeks 7-8)
- Machine learning-enhanced matching
- A/B testing for recommendation effectiveness
- Advanced analytics and reporting
- User feedback integration

## Technical Considerations

### Rate Limiting & Ethics
- Implement respectful crawling delays (1-2 seconds between requests)
- Honor robots.txt and rate limits
- Monitor for bot detection and implement rotation strategies
- Consider legal implications of data scraping

### Error Handling
- Robust retry mechanisms for failed scrapes
- Graceful degradation for parsing failures
- Comprehensive logging for debugging
- Dead letter queues for failed processing

### Scalability
- Asynchronous processing with job queues
- Database indexing for fast matching queries
- Caching strategies for repeated LLM calls
- Horizontal scaling considerations

### LLM Integration
- Prompt engineering for consistent outputs
- Output validation and fallback strategies
- Cost optimization through prompt efficiency
- Model selection based on accuracy vs. cost trade-offs

## Success Metrics

### Technical KPIs
- **Scraping Success Rate**: >95% successful job extractions
- **Parsing Accuracy**: >90% correctly structured outputs
- **Processing Speed**: <30 seconds per job end-to-end
- **System Uptime**: 99.5% availability

### Business KPIs
- **Match Accuracy**: User validation of high-scoring matches
- **Recommendation Effectiveness**: Score improvements after implementing suggestions
- **Application Success Rate**: Interview-to-application ratio for auto-flagged jobs
- **User Engagement**: Resume iteration frequency and improvement adoption

## Tools & Technologies

### Core Stack
- **Web Scraping**: sitemap-extract, crawl4ai
- **LLM Integration**: OpenAI API, Anthropic Claude API
- **Data Processing**: Python (pandas, numpy)
- **Database**: PostgreSQL with full-text search
- **Queue Management**: Redis/Celery for async processing

### Development Tools
- **Version Control**: Git with feature branching
- **Testing**: pytest for unit/integration testing
- **Monitoring**: Logging framework with structured outputs
- **Documentation**: Sphinx for technical documentation

This workflow provides a solid foundation for building an industry-grade career matching system while offering excellent learning opportunities in web scraping, LLM integration, and data processing pipelines.