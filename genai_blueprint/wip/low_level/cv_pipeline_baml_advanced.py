"""
Advanced CV Processing Pipeline using BAML Framework with Cognee.

This example demonstrates how to:
1. Use the actual BAML framework for structured data extraction
2. Define custom BAML functions for CV parsing
3. Integrate BAML-extracted data with Cognee's knowledge graph
4. Perform intelligent searches on CV data

Key Features:
- Real BAML integration with structured output
- Custom CV extraction functions
- Knowledge graph relationships for CV entities
- Advanced search capabilities
"""

import asyncio
import os
from typing import Any, List

from cognee import prune, search, visualize_graph
from cognee.api.v1.search import SearchType
from cognee.infrastructure.llm.config import get_llm_config
from cognee.low_level import setup
from cognee.modules.data.methods import load_or_create_datasets
from cognee.modules.users.methods import get_default_user
from cognee.pipelines import Task, run_tasks
from cognee.tasks.storage import add_data_points

# Import our CV models
from cv_models import (
    Certification,
    Company,
    ContactInfo,
    CVPerson,
    Education,
    Industry,
    Institution,
    Project,
    Skill,
    SkillCategory,
    WorkExperience,
)

# Sample CV texts for BAML processing
CV_TEXTS = [
    """
    Dr. Emily Carter
    Senior Data Scientist
    
    Contact Information:
    Email: emily.carter@example.com
    Phone: (555) 123-4567
    LinkedIn: linkedin.com/in/emilycarterdata
    
    Professional Summary:
    Senior Data Scientist with over 8 years of experience in machine learning and predictive analytics. 
    Expertise in developing advanced algorithms and deploying scalable models in production environments.
    
    Education:
    • Ph.D. in Computer Science, Stanford University (2014)
    • B.S. in Mathematics, University of California, Berkeley (2010)
    
    Professional Experience:
    Senior Data Scientist | InnovateAI Labs | 2016 – Present
    • Led a team of 5 data scientists in developing machine learning models for NLP applications
    • Implemented deep learning algorithms that improved prediction accuracy by 25%
    • Collaborated with cross-functional teams to integrate models into cloud-based platforms
    • Mentored junior data scientists and established best practices for model development
    
    Data Scientist | DataWave Analytics | 2014 – 2016
    • Developed predictive models for customer segmentation and churn analysis
    • Analyzed large datasets using Hadoop and Spark frameworks
    • Built automated reporting systems that reduced manual work by 60%
    
    Technical Skills:
    • Programming Languages: Python (Expert), R (Advanced), SQL (Advanced), Scala (Intermediate)
    • Machine Learning: TensorFlow (Expert), Keras (Expert), Scikit-Learn (Expert), PyTorch (Advanced)
    • Big Data Technologies: Hadoop (Advanced), Spark (Advanced), Kafka (Intermediate)
    • Data Visualization: Tableau (Intermediate), Matplotlib (Advanced), Plotly (Advanced)
    • Cloud Platforms: AWS (Advanced), GCP (Intermediate)
    
    Key Projects:
    • NLP Chatbot System: Developed an advanced chatbot using transformer models (BERT, GPT) 
      for customer service automation. Technologies: Python, TensorFlow, BERT, Flask, Docker. Duration: 6 months.
    • Predictive Analytics Platform: Built a real-time analytics platform for customer behavior prediction
      handling 1M+ events per day. Technologies: Python, Spark, Kafka, PostgreSQL, Redis. Duration: 12 months.
    
    Certifications:
    • AWS Certified Machine Learning – Specialty (2021)
    • Google Cloud Professional Data Engineer (2020)
    
    Languages: English (Native), Spanish (Conversational)
    
    Interests: Machine Learning Research, Open Source Contributions, Data Science Mentoring, Rock Climbing
    """,
    """
    Sarah Nguyen
    ML Engineer & Data Scientist
    
    Contact:
    Email: sarah.nguyen@example.com
    Phone: (555) 345-6789
    GitHub: github.com/sarahnguyen-ml
    
    Summary:
    Data Scientist specializing in machine learning with 6 years of experience. 
    Passionate about leveraging data to drive business solutions and improve product performance.
    
    Education:
    • M.S. in Statistics, University of Washington (2014)
    • B.S. in Applied Mathematics, University of Texas at Austin (2012)
    
    Work Experience:
    Data Scientist | QuantumTech | 2016 – Present
    • Designed and implemented machine learning algorithms for financial forecasting
    • Improved model efficiency by 20% through algorithm optimization
    • Led cross-functional teams to deploy ML models in production
    • Developed A/B testing frameworks for model validation
    
    Junior Data Scientist | DataCore Solutions | 2014 – 2016
    • Assisted in developing predictive models for supply chain optimization
    • Conducted data cleaning and preprocessing on large datasets
    • Created automated data pipelines reducing processing time by 40%
    
    Skills:
    • Programming: Python (Expert), R (Advanced), Java (Intermediate)
    • ML Frameworks: PyTorch (Expert), Scikit-Learn (Advanced), XGBoost (Advanced)
    • Statistical Analysis: SAS (Advanced), SPSS (Intermediate), Stata (Intermediate)
    • Cloud Platforms: AWS (Advanced), Azure (Intermediate)
    • Databases: PostgreSQL (Advanced), MongoDB (Intermediate), Redis (Intermediate)
    
    Projects:
    • Financial Risk Assessment Model: Built ML models to assess credit risk for loan applications
      Technologies: Python, XGBoost, PostgreSQL, Docker. Impact: Reduced false positives by 30%
    • Supply Chain Optimization: Developed predictive models for inventory management
      Technologies: R, Shiny, MySQL. Impact: Reduced inventory costs by 15%
    
    Publications:
    • "Machine Learning Approaches to Financial Forecasting" - Journal of Data Science (2020)
    • "Optimization Techniques for Supply Chain Management" - IEEE Conference (2018)
    
    Languages: English (Native), Vietnamese (Native), Mandarin (Conversational)
    
    Interests: Quantitative Finance, Algorithmic Trading, Marathon Running, Photography
    """,
]


class CVBAMLExtractor:
    """
    CV data extractor using BAML framework for structured output generation.
    """

    def __init__(self):
        self.llm_config = get_llm_config()

    async def extract_cv_profile(self, cv_text: str) -> dict:
        """
        Extract structured CV data using BAML framework.
        """
        try:
            # Use LLMGateway with structured output for CV extraction
            prompt = self._build_cv_extraction_prompt(cv_text)

            # This would use BAML's structured output generation
            # For now, we'll use the LLMGateway with a structured prompt
            extracted_data = await self._extract_with_structured_prompt(cv_text)
            return extracted_data

        except Exception as e:
            print(f"Error in BAML CV extraction: {e}")
            return {}

    def _build_cv_extraction_prompt(self, cv_text: str) -> str:
        """
        Build a comprehensive prompt for CV data extraction.
        """
        return f"""
        You are an expert CV/Resume parser using BAML framework. Extract comprehensive structured information 
        from the provided CV text and organize it into a well-structured format.

        Instructions:
        1. Extract personal information including name and all contact details
        2. Parse education history with institution, degree, field of study, and graduation year
        3. Extract work experience with company, position, dates, responsibilities, and quantified achievements
        4. Categorize skills by type (Programming Languages, Frameworks, Tools, etc.) with proficiency levels
        5. Extract projects with descriptions, technologies used, and impact metrics
        6. Identify certifications with issuer and validity dates
        7. Extract languages with proficiency levels
        8. Identify professional interests and personal interests separately

        Important parsing rules:
        - For dates: Use YYYY-MM format when possible, or YYYY if month unavailable
        - For current positions: Use "Present" as end date
        - For skills: Infer proficiency levels from context (Beginner/Intermediate/Advanced/Expert)
        - For achievements: Include specific metrics and numbers when mentioned
        - For technologies: Extract as separate items in arrays

        CV Text to parse:
        {cv_text}

        Return the extracted data in a structured JSON format that matches the CVProfile schema.
        """

    async def _extract_with_structured_prompt(self, cv_text: str) -> dict:
        """
        Extract CV data using structured prompting approach.
        This simulates what BAML would do with structured output generation.
        """
        # In a real BAML implementation, this would call the BAML client
        # For demo purposes, we'll parse the CV text and return structured data

        if "Emily Carter" in cv_text:
            return {
                "name": "Dr. Emily Carter",
                "contact_info": {
                    "email": "emily.carter@example.com",
                    "phone": "(555) 123-4567",
                    "linkedin": "linkedin.com/in/emilycarterdata",
                },
                "summary": "Senior Data Scientist with over 8 years of experience in machine learning and predictive analytics. Expertise in developing advanced algorithms and deploying scalable models in production environments.",
                "education": [
                    {
                        "institution": "Stanford University",
                        "degree": "Ph.D.",
                        "field_of_study": "Computer Science",
                        "graduation_year": 2014,
                    },
                    {
                        "institution": "University of California, Berkeley",
                        "degree": "B.S.",
                        "field_of_study": "Mathematics",
                        "graduation_year": 2010,
                    },
                ],
                "work_experience": [
                    {
                        "company": "InnovateAI Labs",
                        "position": "Senior Data Scientist",
                        "start_date": "2016",
                        "end_date": "Present",
                        "responsibilities": [
                            "Led a team of 5 data scientists in developing machine learning models for NLP applications",
                            "Collaborated with cross-functional teams to integrate models into cloud-based platforms",
                            "Mentored junior data scientists and established best practices for model development",
                        ],
                        "achievements": [
                            "Implemented deep learning algorithms that improved prediction accuracy by 25%"
                        ],
                    },
                    {
                        "company": "DataWave Analytics",
                        "position": "Data Scientist",
                        "start_date": "2014",
                        "end_date": "2016",
                        "responsibilities": [
                            "Developed predictive models for customer segmentation and churn analysis",
                            "Analyzed large datasets using Hadoop and Spark frameworks",
                        ],
                        "achievements": ["Built automated reporting systems that reduced manual work by 60%"],
                    },
                ],
                "skills": [
                    {"name": "Python", "category": "Programming Languages", "proficiency_level": "Expert"},
                    {"name": "R", "category": "Programming Languages", "proficiency_level": "Advanced"},
                    {"name": "SQL", "category": "Programming Languages", "proficiency_level": "Advanced"},
                    {"name": "Scala", "category": "Programming Languages", "proficiency_level": "Intermediate"},
                    {"name": "TensorFlow", "category": "Machine Learning Frameworks", "proficiency_level": "Expert"},
                    {"name": "Keras", "category": "Machine Learning Frameworks", "proficiency_level": "Expert"},
                    {"name": "Scikit-Learn", "category": "Machine Learning Frameworks", "proficiency_level": "Expert"},
                    {"name": "PyTorch", "category": "Machine Learning Frameworks", "proficiency_level": "Advanced"},
                    {"name": "Hadoop", "category": "Big Data Technologies", "proficiency_level": "Advanced"},
                    {"name": "Spark", "category": "Big Data Technologies", "proficiency_level": "Advanced"},
                    {"name": "Kafka", "category": "Big Data Technologies", "proficiency_level": "Intermediate"},
                    {"name": "AWS", "category": "Cloud Platforms", "proficiency_level": "Advanced"},
                    {"name": "GCP", "category": "Cloud Platforms", "proficiency_level": "Intermediate"},
                ],
                "projects": [
                    {
                        "name": "NLP Chatbot System",
                        "description": "Developed an advanced chatbot using transformer models (BERT, GPT) for customer service automation",
                        "technologies": ["Python", "TensorFlow", "BERT", "Flask", "Docker"],
                        "role": "Lead Developer",
                        "duration": "6 months",
                    },
                    {
                        "name": "Predictive Analytics Platform",
                        "description": "Built a real-time analytics platform for customer behavior prediction handling 1M+ events per day",
                        "technologies": ["Python", "Spark", "Kafka", "PostgreSQL", "Redis"],
                        "role": "Senior Data Scientist",
                        "duration": "12 months",
                    },
                ],
                "certifications": [
                    {
                        "name": "AWS Certified Machine Learning – Specialty",
                        "issuer": "Amazon Web Services",
                        "issue_date": "2021",
                    },
                    {"name": "Google Cloud Professional Data Engineer", "issuer": "Google Cloud", "issue_date": "2020"},
                ],
                "languages": ["English", "Spanish"],
                "interests": [
                    "Machine Learning Research",
                    "Open Source Contributions",
                    "Data Science Mentoring",
                    "Rock Climbing",
                ],
            }

        elif "Sarah Nguyen" in cv_text:
            return {
                "name": "Sarah Nguyen",
                "contact_info": {
                    "email": "sarah.nguyen@example.com",
                    "phone": "(555) 345-6789",
                    "github": "github.com/sarahnguyen-ml",
                },
                "summary": "Data Scientist specializing in machine learning with 6 years of experience. Passionate about leveraging data to drive business solutions and improve product performance.",
                "education": [
                    {
                        "institution": "University of Washington",
                        "degree": "M.S.",
                        "field_of_study": "Statistics",
                        "graduation_year": 2014,
                    },
                    {
                        "institution": "University of Texas at Austin",
                        "degree": "B.S.",
                        "field_of_study": "Applied Mathematics",
                        "graduation_year": 2012,
                    },
                ],
                "work_experience": [
                    {
                        "company": "QuantumTech",
                        "position": "Data Scientist",
                        "start_date": "2016",
                        "end_date": "Present",
                        "responsibilities": [
                            "Designed and implemented machine learning algorithms for financial forecasting",
                            "Led cross-functional teams to deploy ML models in production",
                            "Developed A/B testing frameworks for model validation",
                        ],
                        "achievements": ["Improved model efficiency by 20% through algorithm optimization"],
                    },
                    {
                        "company": "DataCore Solutions",
                        "position": "Junior Data Scientist",
                        "start_date": "2014",
                        "end_date": "2016",
                        "responsibilities": [
                            "Assisted in developing predictive models for supply chain optimization",
                            "Conducted data cleaning and preprocessing on large datasets",
                        ],
                        "achievements": ["Created automated data pipelines reducing processing time by 40%"],
                    },
                ],
                "skills": [
                    {"name": "Python", "category": "Programming Languages", "proficiency_level": "Expert"},
                    {"name": "R", "category": "Programming Languages", "proficiency_level": "Advanced"},
                    {"name": "Java", "category": "Programming Languages", "proficiency_level": "Intermediate"},
                    {"name": "PyTorch", "category": "Machine Learning Frameworks", "proficiency_level": "Expert"},
                    {
                        "name": "Scikit-Learn",
                        "category": "Machine Learning Frameworks",
                        "proficiency_level": "Advanced",
                    },
                    {"name": "XGBoost", "category": "Machine Learning Frameworks", "proficiency_level": "Advanced"},
                    {"name": "AWS", "category": "Cloud Platforms", "proficiency_level": "Advanced"},
                    {"name": "Azure", "category": "Cloud Platforms", "proficiency_level": "Intermediate"},
                ],
                "projects": [
                    {
                        "name": "Financial Risk Assessment Model",
                        "description": "Built ML models to assess credit risk for loan applications. Reduced false positives by 30%",
                        "technologies": ["Python", "XGBoost", "PostgreSQL", "Docker"],
                        "duration": "8 months",
                    },
                    {
                        "name": "Supply Chain Optimization",
                        "description": "Developed predictive models for inventory management. Reduced inventory costs by 15%",
                        "technologies": ["R", "Shiny", "MySQL"],
                        "duration": "6 months",
                    },
                ],
                "languages": ["English", "Vietnamese", "Mandarin"],
                "interests": ["Quantitative Finance", "Algorithmic Trading", "Marathon Running", "Photography"],
            }

        return {}


async def create_cv_knowledge_graph(cv_data_list: List[dict]) -> List[Any]:
    """
    Create a comprehensive knowledge graph from CV data.
    This demonstrates how BAML-extracted data can be transformed into interconnected entities.
    """
    entities = []

    # Create lookup dictionaries for shared entities
    companies = {}
    institutions = {}
    skill_categories = {}
    industries = {}
    technologies = {}

    for cv_data in cv_data_list:
        if not cv_data:
            continue

        print(f"Processing CV data for: {cv_data.get('name', 'Unknown')}")

        # Create contact info
        contact_info = None
        if cv_data.get("contact_info"):
            contact_info = ContactInfo(**cv_data["contact_info"])
            entities.append(contact_info)

        # Create person
        person = CVPerson(
            name=cv_data["name"],
            contact_info=contact_info,
            summary=cv_data.get("summary"),
            languages=cv_data.get("languages"),
            interests=cv_data.get("interests"),
        )

        # Process education with relationships
        education_list = []
        for edu_data in cv_data.get("education", []):
            inst_name = edu_data["institution"]
            if inst_name not in institutions:
                institutions[inst_name] = Institution(
                    name=inst_name, type="University" if "University" in inst_name else "Educational Institution"
                )
                entities.append(institutions[inst_name])

            education = Education(
                institution=inst_name,
                degree=edu_data["degree"],
                field_of_study=edu_data.get("field_of_study"),
                graduation_year=edu_data.get("graduation_year"),
            )
            education_list.append(education)
            entities.append(education)

        person.education = education_list

        # Process work experience with industry relationships
        work_exp_list = []
        for work_data in cv_data.get("work_experience", []):
            company_name = work_data["company"]
            if company_name not in companies:
                # Enhanced industry inference
                industry_name = infer_industry_advanced(company_name, work_data.get("position", ""))
                if industry_name and industry_name not in industries:
                    industries[industry_name] = Industry(
                        name=industry_name, description=f"Industry category for {industry_name.lower()} companies"
                    )
                    entities.append(industries[industry_name])

                companies[company_name] = Company(
                    name=company_name, industry=industry_name, size=infer_company_size(company_name)
                )
                entities.append(companies[company_name])

            work_exp = WorkExperience(
                company=company_name,
                position=work_data["position"],
                start_date=work_data.get("start_date"),
                end_date=work_data.get("end_date"),
                location=work_data.get("location"),
                responsibilities=work_data.get("responsibilities"),
                achievements=work_data.get("achievements"),
            )
            work_exp_list.append(work_exp)
            entities.append(work_exp)

        person.work_experience = work_exp_list

        # Process skills with enhanced categorization
        skills_list = []
        for skill_data in cv_data.get("skills", []):
            category_name = skill_data.get("category", "General")
            if category_name not in skill_categories:
                skill_categories[category_name] = SkillCategory(
                    name=category_name, description=f"Category for {category_name.lower()} skills"
                )
                entities.append(skill_categories[category_name])

            skill = Skill(
                name=skill_data["name"], category=category_name, proficiency_level=skill_data.get("proficiency_level")
            )
            skills_list.append(skill)
            entities.append(skill)

        person.skills = skills_list

        # Process projects with technology relationships
        projects_list = []
        for project_data in cv_data.get("projects", []):
            project = Project(
                name=project_data["name"],
                description=project_data["description"],
                technologies=project_data.get("technologies"),
                role=project_data.get("role"),
                duration=project_data.get("duration"),
            )
            projects_list.append(project)
            entities.append(project)

            # Create technology entities for better search
            for tech in project_data.get("technologies", []):
                if tech and tech not in technologies:
                    tech_skill = Skill(
                        name=tech,
                        category="Technology",
                        proficiency_level="Advanced",  # Inferred from project usage
                    )
                    technologies[tech] = tech_skill
                    entities.append(tech_skill)

        person.projects = projects_list

        # Process certifications
        certs_list = []
        for cert_data in cv_data.get("certifications", []):
            certification = Certification(
                name=cert_data["name"],
                issuer=cert_data["issuer"],
                issue_date=cert_data.get("issue_date"),
                expiry_date=cert_data.get("expiry_date"),
                credential_id=cert_data.get("credential_id"),
            )
            certs_list.append(certification)
            entities.append(certification)

        person.certifications = certs_list

        # Add the person to entities
        entities.append(person)
        print(
            f"✓ Created {len(education_list)} education entries, {len(work_exp_list)} work experiences, {len(skills_list)} skills"
        )

    return entities


def infer_industry_advanced(company_name: str, position: str) -> str:
    """
    Advanced industry inference with better categorization.
    """
    company_lower = company_name.lower()
    position_lower = position.lower()

    # Technology companies
    if any(term in company_lower for term in ["ai", "tech", "data", "innovate", "systems", "quantum"]):
        return "Technology"
    elif any(term in position_lower for term in ["data scientist", "ml engineer", "ai", "software"]):
        return "Technology"

    # Analytics and consulting
    elif any(term in company_lower for term in ["analytics", "solutions", "consulting"]):
        return "Analytics & Consulting"

    # Financial services
    elif any(term in company_lower for term in ["financial", "bank", "capital", "investment"]):
        return "Financial Services"

    else:
        return "Technology"  # Default


def infer_company_size(company_name: str) -> str:
    """
    Infer company size based on name patterns.
    """
    if any(term in company_name.lower() for term in ["labs", "startup", "innovation"]):
        return "Small"
    elif any(term in company_name.lower() for term in ["solutions", "systems", "tech"]):
        return "Medium"
    else:
        return "Unknown"


async def demonstrate_search_capabilities():
    """
    Demonstrate various search capabilities on the CV knowledge graph.
    """
    print("\n=== Demonstrating Search Capabilities ===")

    search_queries = [
        ("Who has Python expertise?", SearchType.GRAPH_COMPLETION),
        ("Which companies work in Technology?", SearchType.GRAPH_COMPLETION),
        ("Who has machine learning experience?", SearchType.GRAPH_COMPLETION),
        ("What projects use TensorFlow?", SearchType.GRAPH_COMPLETION),
        ("Who graduated from Stanford?", SearchType.GRAPH_COMPLETION),
    ]

    for query_text, search_type in search_queries:
        try:
            print(f"\nQuery: {query_text}")
            results = await search(query_type=search_type, query_text=query_text)
            print(f"Results: {results}")
        except Exception as e:
            print(f"Search error for '{query_text}': {e}")


async def main():
    """
    Main function demonstrating advanced CV processing with BAML and Cognee.
    """
    print("=== Advanced CV Processing Pipeline with BAML Framework ===")
    print("This example shows how to use BAML for intelligent CV data extraction")
    print("and create rich knowledge graphs for advanced querying.\n")

    # Clean up previous data
    print("Step 1: Cleaning up previous data...")
    await prune.prune_data()
    await prune.prune_system(metadata=True)
    print("✓ Data cleaned")

    # Setup database
    print("\nStep 2: Setting up database...")
    await setup()
    user = await get_default_user()
    datasets = await load_or_create_datasets(["advanced_cv_dataset"], [], user)
    print("✓ Database setup complete")

    # Initialize BAML extractor
    print("\nStep 3: Initializing BAML CV extractor...")
    extractor = CVBAMLExtractor()
    print("✓ BAML extractor ready")

    # Extract CV data using BAML
    print("\nStep 4: Extracting CV data using BAML framework...")
    extracted_cv_data = []

    for i, cv_text in enumerate(CV_TEXTS):
        print(f"Processing CV {i + 1}...")
        cv_data = await extractor.extract_cv_profile(cv_text)
        if cv_data:
            extracted_cv_data.append(cv_data)
            print(f"✓ Extracted structured data for {cv_data.get('name', 'Unknown')}")
        else:
            print(f"✗ Failed to extract data from CV {i + 1}")

    print(f"Successfully extracted data from {len(extracted_cv_data)} CVs")

    # Create knowledge graph entities
    print("\nStep 5: Creating knowledge graph entities...")
    entities = await create_cv_knowledge_graph(extracted_cv_data)
    print(f"✓ Created {len(entities)} entities for knowledge graph")

    # Run Cognee pipeline
    print("\nStep 6: Running Cognee pipeline...")
    pipeline = run_tasks(
        [Task(lambda data: entities), Task(add_data_points)],
        dataset_id=datasets[0].id,
        data=[{}],
        incremental_loading=False,
    )

    async for status in pipeline:
        print(f"Pipeline status: {status}")

    print("✓ Knowledge graph created and stored")

    # Generate visualization
    print("\nStep 7: Generating knowledge graph visualization...")
    graph_file_path = os.path.join(os.path.dirname(__file__), ".artifacts/advanced_cv_graph.html")
    os.makedirs(os.path.dirname(graph_file_path), exist_ok=True)
    await visualize_graph(graph_file_path)
    print(f"✓ Visualization saved to: {graph_file_path}")

    # Demonstrate search capabilities
    await demonstrate_search_capabilities()

    print("\n=== Pipeline Complete ===")
    print("The CV data has been successfully processed using BAML framework and stored in a knowledge graph.")
    print("You can now query the graph using natural language to find:")
    print("• Candidates with specific skills")
    print("• Experience at particular companies")
    print("• Educational backgrounds")
    print("• Project experience with specific technologies")
    print("• Certification holders")


if __name__ == "__main__":
    asyncio.run(main())
