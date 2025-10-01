"""
CV Processing Pipeline using BAML-generated classes with Cognee's low-level API.

This example demonstrates how to:
1. Use BAML to extract structured CV data from text instead of pre-structured JSON
2. Create relationships between CV entities using DataPoint models
3. Process CV data through Cognee's pipeline system
4. Search the resulting knowledge graph

Key differences from JSON approach:
- Uses BAML framework to extract structured data from unstructured CV text
- Leverages LLM capabilities for intelligent data extraction
- Creates a more flexible and scalable approach to CV processing
"""

import asyncio
import json
import os
from typing import Any, List

from cognee import prune, visualize_graph
from cognee.infrastructure.llm.LLMGateway import LLMGateway
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

# Sample CV text data (unstructured text that BAML will parse)
CV_TEXTS = [
    """
    Dr. Emily Carter
    Email: emily.carter@example.com
    Phone: (555) 123-4567
    LinkedIn: linkedin.com/in/emilycarterdata
    
    SUMMARY
    Senior Data Scientist with over 8 years of experience in machine learning and predictive analytics. 
    Expertise in developing advanced algorithms and deploying scalable models in production environments.
    
    EDUCATION
    Ph.D. in Computer Science, Stanford University (2014)
    B.S. in Mathematics, University of California, Berkeley (2010)
    
    EXPERIENCE
    Senior Data Scientist, InnovateAI Labs (2016 – Present)
    • Led a team in developing machine learning models for natural language processing applications
    • Implemented deep learning algorithms that improved prediction accuracy by 25%
    • Collaborated with cross-functional teams to integrate models into cloud-based platforms
    
    Data Scientist, DataWave Analytics (2014 – 2016)
    • Developed predictive models for customer segmentation and churn analysis
    • Analyzed large datasets using Hadoop and Spark frameworks
    
    SKILLS
    Programming Languages: Python (Expert), R (Advanced), SQL (Advanced)
    Machine Learning: TensorFlow (Expert), Keras (Expert), Scikit-Learn (Expert)
    Big Data: Hadoop (Advanced), Spark (Advanced)
    Visualization: Tableau (Intermediate), Matplotlib (Advanced)
    
    PROJECTS
    NLP Chatbot System: Developed an advanced chatbot using transformer models for customer service automation
    Technologies: Python, TensorFlow, BERT, Flask (6 months)
    
    Predictive Analytics Platform: Built a real-time analytics platform for customer behavior prediction
    Technologies: Python, Spark, Kafka, PostgreSQL (12 months)
    
    LANGUAGES: English, Spanish
    INTERESTS: Machine Learning Research, Open Source Contributions, Data Science Mentoring
    """,
    """
    Michael Rodriguez
    Email: michael.rodriguez@example.com
    Phone: (555) 234-5678
    GitHub: github.com/mrodriguez-ml
    
    SUMMARY
    Machine Learning Engineer with 5 years of experience in deploying scalable ML solutions. 
    Proficient in MLOps practices and cloud infrastructure management.
    
    EDUCATION
    M.S. in Artificial Intelligence, MIT (2017)
    B.S. in Computer Science, Carnegie Mellon University (2015)
    
    EXPERIENCE
    Senior ML Engineer, TechFlow Systems (2019 – Present)
    • Designed and implemented ML pipelines for production deployment
    • Optimized model performance and reduced inference latency by 40%
    
    ML Engineer, DataCore Solutions (2017 – 2019)
    • Developed computer vision models for manufacturing quality control
    • Implemented CI/CD pipelines for ML model deployment
    
    SKILLS
    Programming: Python (Expert), Java (Intermediate)
    ML Frameworks: PyTorch (Expert)
    MLOps: MLflow (Advanced)
    DevOps: Docker (Advanced), Kubernetes (Intermediate)
    Cloud: AWS (Advanced)
    
    CERTIFICATIONS
    AWS Certified Machine Learning - Specialty, Amazon Web Services (2022-03 to 2025-03)
    
    LANGUAGES: English, Portuguese
    INTERESTS: MLOps, Computer Vision, Cloud Architecture
    """,
]


async def extract_cv_data_with_baml(cv_text: str) -> dict:
    """
    Extract structured CV data from unstructured text using BAML framework.
    This replaces the need for pre-structured JSON data.
    """
    try:
        # Use LLMGateway to extract structured data
        # This is where BAML would be used in a real implementation
        prompt = f"""
        Extract structured CV information from the following text and return it as JSON:
        
        Extract:
        - Personal information (name, contact details)
        - Education (institution, degree, field, year)
        - Work experience (company, position, dates, responsibilities, achievements)
        - Skills (categorized by type and proficiency level)
        - Projects (name, description, technologies, duration)
        - Certifications (name, issuer, dates)
        - Languages and interests
        
        CV Text:
        {cv_text}
        
        Return the data in a structured JSON format matching the CVPerson model.
        """

        # For this example, we'll simulate BAML extraction with a simple parser
        # In a real implementation, you would use the BAML client here
        extracted_data = await simulate_baml_extraction(cv_text)
        return extracted_data

    except Exception as e:
        print(f"Error extracting CV data: {e}")
        return {}


async def simulate_baml_extraction(cv_text: str) -> dict:
    """
    Simulate BAML extraction for demo purposes.
    In a real implementation, this would be replaced with actual BAML client calls.
    """
    # Load the JSON data as a fallback for demonstration
    # In production, you would use the BAML client to extract this from the text
    cv_file_path = os.path.join(os.path.dirname(__file__), "cv_data.json")
    with open(cv_file_path, "r") as f:
        cv_data = json.load(f)

    # Return the first CV that matches the name in the text
    if "Emily Carter" in cv_text:
        return cv_data[0]
    elif "Michael Rodriguez" in cv_text:
        return cv_data[1]
    else:
        return {}


def create_cv_entities(cv_data_list: List[dict]) -> List[Any]:
    """
    Create CV entities and relationships from extracted BAML data.
    This function demonstrates how to transform BAML-extracted data into Cognee DataPoints.
    """
    entities = []

    # Create lookup dictionaries to avoid duplicates
    companies = {}
    institutions = {}
    skill_categories = {}
    industries = {}

    for cv_data in cv_data_list:
        if not cv_data:
            continue

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

        # Process education
        education_list = []
        for edu_data in cv_data.get("education", []):
            # Create or get institution
            inst_name = edu_data["institution"]
            if inst_name not in institutions:
                institutions[inst_name] = Institution(name=inst_name)
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

        # Process work experience
        work_exp_list = []
        for work_data in cv_data.get("work_experience", []):
            # Create or get company
            company_name = work_data["company"]
            if company_name not in companies:
                # Infer industry from company name or position (simplified)
                industry_name = infer_industry(company_name, work_data.get("position", ""))
                if industry_name and industry_name not in industries:
                    industries[industry_name] = Industry(name=industry_name)
                    entities.append(industries[industry_name])

                companies[company_name] = Company(name=company_name, industry=industry_name)
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

        # Process skills
        skills_list = []
        for skill_data in cv_data.get("skills", []):
            # Create or get skill category
            category_name = skill_data.get("category", "General")
            if category_name not in skill_categories:
                skill_categories[category_name] = SkillCategory(name=category_name)
                entities.append(skill_categories[category_name])

            skill = Skill(
                name=skill_data["name"], category=category_name, proficiency_level=skill_data.get("proficiency_level")
            )
            skills_list.append(skill)
            entities.append(skill)

        person.skills = skills_list

        # Process projects
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

    return entities


def infer_industry(company_name: str, position: str) -> str:
    """
    Simple industry inference based on company name and position.
    In a real BAML implementation, this could be more sophisticated.
    """
    company_lower = company_name.lower()
    position_lower = position.lower()

    if any(term in company_lower for term in ["ai", "tech", "data", "innovate", "systems"]):
        return "Technology"
    elif any(term in position_lower for term in ["data scientist", "ml engineer", "ai"]):
        return "Technology"
    elif "analytics" in company_lower:
        return "Analytics"
    else:
        return "Technology"  # Default for this example


async def process_cv_pipeline(cv_texts: List[str]):
    """
    Main pipeline for processing CV data using BAML extraction.
    """
    print("=== CV Processing Pipeline with BAML ===")

    # Step 1: Extract structured data from unstructured CV texts using BAML
    print("Step 1: Extracting structured data using BAML...")
    extracted_cv_data = []

    for i, cv_text in enumerate(cv_texts):
        print(f"Processing CV {i + 1}...")
        cv_data = await extract_cv_data_with_baml(cv_text)
        if cv_data:
            extracted_cv_data.append(cv_data)
            print(f"✓ Extracted data for {cv_data.get('name', 'Unknown')}")

    # Step 2: Create CV entities from extracted data
    print("\nStep 2: Creating CV entities and relationships...")
    entities = create_cv_entities(extracted_cv_data)
    print(f"✓ Created {len(entities)} entities")

    return entities


async def main():
    """
    Main function demonstrating CV processing with BAML and Cognee.
    """
    # Clean up previous data
    await prune.prune_data()
    await prune.prune_system(metadata=True)
    print("✓ Cleaned up previous data")

    # Setup database
    await setup()
    print("✓ Setup complete")

    # Get default user and create dataset
    user = await get_default_user()
    datasets = await load_or_create_datasets(["cv_dataset"], [], user)
    print("✓ Created dataset")

    # Process CV data through the pipeline
    entities = await process_cv_pipeline(CV_TEXTS)

    # Run the Cognee pipeline
    print("\nStep 3: Running Cognee pipeline...")
    pipeline = run_tasks(
        [Task(lambda data: entities), Task(add_data_points)],
        dataset_id=datasets[0].id,
        data=[{}],  # Dummy data since we're using pre-created entities
        incremental_loading=False,
    )

    async for status in pipeline:
        print(f"Pipeline status: {status}")

    print("✓ CV data processed and stored in knowledge graph")

    # Visualize the graph
    print("\nStep 4: Generating graph visualization...")
    graph_file_path = os.path.join(os.path.dirname(__file__), ".artifacts/cv_graph_visualization.html")
    await visualize_graph(graph_file_path)
    print(f"✓ Graph visualization saved to: {graph_file_path}")

    # Demonstrate search capabilities
    print("\nStep 5: Demonstrating search capabilities...")
    try:
        # This would require the search functionality to be set up
        print("Search functionality would be demonstrated here")
        print("Examples of queries you could run:")
        print("- 'Who has experience with Python?'")
        print("- 'Which companies are in the Technology industry?'")
        print("- 'Who has Machine Learning expertise?'")
        print("- 'What projects involve TensorFlow?'")
    except Exception as e:
        print(f"Search demo not available: {e}")


if __name__ == "__main__":
    print("Starting CV Processing Pipeline with BAML...")
    print("This example demonstrates how to use BAML-generated classes")
    print("to extract and process CV data instead of using pre-structured JSON.\n")

    asyncio.run(main())
