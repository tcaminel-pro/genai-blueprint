"""
Comparison between JSON and BAML approaches for CV processing.

This script demonstrates the key differences between:
1. Traditional JSON-based CV processing (like the original pipeline.py)
2. BAML-based intelligent CV extraction

Key differences highlighted:
- Data input format (structured vs unstructured)
- Extraction method (parsing vs LLM-powered)
- Flexibility and adaptability
- Scalability considerations
"""

import asyncio
import json
import os
from typing import Any, Dict, List

from cognee import prune
from cognee.low_level import DataPoint, setup
from cognee.modules.data.methods import load_or_create_datasets
from cognee.modules.users.methods import get_default_user
from cognee.pipelines import Task, run_tasks
from cognee.tasks.storage import add_data_points


# Traditional JSON approach (similar to original pipeline.py)
class PersonJSON(DataPoint):
    name: str
    company: str
    department: str
    metadata: dict = {"index_fields": ["name", "company", "department"]}


class CompanyJSON(DataPoint):
    name: str
    industry: str
    metadata: dict = {"index_fields": ["name", "industry"]}


# BAML approach classes (simplified version)
class CVPersonBAML(DataPoint):
    name: str
    current_company: str
    skills: List[str]
    experience_years: int
    education_level: str
    summary: str
    metadata: dict = {"index_fields": ["name", "skills", "summary"]}


class SkillBAML(DataPoint):
    name: str
    category: str
    proficiency: str
    metadata: dict = {"index_fields": ["name", "category"]}


# Sample data for comparison
JSON_SAMPLE_DATA = [
    {
        "people": [
            {"name": "John Doe", "company": "TechNova Inc.", "department": "Engineering"},
            {"name": "Jane Smith", "company": "TechNova Inc.", "department": "Marketing"},
        ],
        "companies": [{"name": "TechNova Inc.", "industry": "Technology"}],
    }
]

UNSTRUCTURED_CV_SAMPLE = """
John Doe
Senior Software Engineer at TechNova Inc.

Summary: Experienced software engineer with 8 years in full-stack development. 
Specialized in Python, JavaScript, and cloud technologies. Led multiple high-impact 
projects and mentored junior developers.

Experience:
- Senior Software Engineer at TechNova Inc. (2020-Present)
- Software Developer at StartupXYZ (2016-2020)

Education: B.S. Computer Science, MIT (2016)

Skills: Python (Expert), JavaScript (Advanced), AWS (Advanced), 
Docker (Intermediate), Machine Learning (Beginner)

Notable Achievements:
- Led migration to microservices architecture
- Reduced system latency by 40%
- Mentored 5 junior developers
"""


def process_json_approach(data: List[Dict]) -> List[DataPoint]:
    """
    Traditional JSON approach: Parse pre-structured data.

    Pros:
    - Fast and reliable
    - No LLM costs
    - Consistent structure

    Cons:
    - Requires manual data structuring
    - Limited to predefined schema
    - No intelligent extraction
    """
    entities = []

    for data_item in data:
        # Process companies
        for company_data in data_item.get("companies", []):
            company = CompanyJSON(name=company_data["name"], industry=company_data.get("industry", "Unknown"))
            entities.append(company)

        # Process people
        for person_data in data_item.get("people", []):
            person = PersonJSON(
                name=person_data["name"], company=person_data["company"], department=person_data["department"]
            )
            entities.append(person)

    return entities


async def process_baml_approach(cv_text: str) -> List[DataPoint]:
    """
    BAML approach: Extract structured data from unstructured text.

    Pros:
    - Handles any CV format
    - Intelligent extraction
    - Rich data extraction
    - Adaptive to new information

    Cons:
    - Requires LLM calls
    - Potential for extraction errors
    - More complex setup
    """
    entities = []

    # Simulate BAML extraction (in real implementation, this would use BAML client)
    extracted_data = await simulate_intelligent_extraction(cv_text)

    if extracted_data:
        # Create person entity with rich information
        person = CVPersonBAML(
            name=extracted_data["name"],
            current_company=extracted_data["current_company"],
            skills=extracted_data["skills"],
            experience_years=extracted_data["experience_years"],
            education_level=extracted_data["education_level"],
            summary=extracted_data["summary"],
        )
        entities.append(person)

        # Create skill entities
        for skill_name, details in extracted_data["skill_details"].items():
            skill = SkillBAML(name=skill_name, category=details["category"], proficiency=details["proficiency"])
            entities.append(skill)

    return entities


async def simulate_intelligent_extraction(cv_text: str) -> Dict[str, Any]:
    """
    Simulate what BAML would extract from unstructured CV text.
    In a real implementation, this would be a BAML client call.
    """
    # This simulates intelligent extraction that BAML would perform
    if "John Doe" in cv_text:
        return {
            "name": "John Doe",
            "current_company": "TechNova Inc.",
            "skills": ["Python", "JavaScript", "AWS", "Docker", "Machine Learning"],
            "experience_years": 8,
            "education_level": "Bachelor's Degree",
            "summary": "Experienced software engineer with 8 years in full-stack development. Specialized in Python, JavaScript, and cloud technologies.",
            "skill_details": {
                "Python": {"category": "Programming Languages", "proficiency": "Expert"},
                "JavaScript": {"category": "Programming Languages", "proficiency": "Advanced"},
                "AWS": {"category": "Cloud Platforms", "proficiency": "Advanced"},
                "Docker": {"category": "DevOps Tools", "proficiency": "Intermediate"},
                "Machine Learning": {"category": "AI/ML", "proficiency": "Beginner"},
            },
        }
    return {}


async def run_comparison():
    """
    Run both approaches and compare results.
    """
    print("=== JSON vs BAML Approach Comparison ===\n")

    # Setup
    await prune.prune_data()
    await prune.prune_system(metadata=True)
    await setup()
    user = await get_default_user()

    # JSON Approach
    print("1. JSON APPROACH")
    print("================")
    print("Input: Pre-structured JSON data")
    print("Processing method: Direct parsing")
    print("Schema: Fixed, predefined")

    datasets_json = await load_or_create_datasets(["json_approach"], [], user)
    json_entities = process_json_approach(JSON_SAMPLE_DATA)

    print(f"Entities created: {len(json_entities)}")
    for entity in json_entities:
        print(f"  - {type(entity).__name__}: {entity.name}")

    # Run JSON pipeline
    pipeline_json = run_tasks(
        [Task(lambda data: json_entities), Task(add_data_points)],
        dataset_id=datasets_json[0].id,
        data=[{}],
        incremental_loading=False,
    )

    json_statuses = []
    async for status in pipeline_json:
        json_statuses.append(status)

    print(f"Pipeline completed with {len(json_statuses)} steps\n")

    # BAML Approach
    print("2. BAML APPROACH")
    print("================")
    print("Input: Unstructured CV text")
    print("Processing method: LLM-powered extraction")
    print("Schema: Adaptive, intelligent")

    datasets_baml = await load_or_create_datasets(["baml_approach"], [], user)
    baml_entities = await process_baml_approach(UNSTRUCTURED_CV_SAMPLE)

    print(f"Entities created: {len(baml_entities)}")
    for entity in baml_entities:
        if hasattr(entity, "name"):
            print(f"  - {type(entity).__name__}: {entity.name}")
        else:
            print(f"  - {type(entity).__name__}")

    # Run BAML pipeline
    pipeline_baml = run_tasks(
        [Task(lambda data: baml_entities), Task(add_data_points)],
        dataset_id=datasets_baml[0].id,
        data=[{}],
        incremental_loading=False,
    )

    baml_statuses = []
    async for status in pipeline_baml:
        baml_statuses.append(status)

    print(f"Pipeline completed with {len(baml_statuses)} steps\n")

    # Comparison Analysis
    print("3. DETAILED COMPARISON")
    print("======================")

    print("Data Richness:")
    print(f"  JSON Approach: {len(json_entities)} entities with basic fields")
    print(f"  BAML Approach: {len(baml_entities)} entities with rich information")

    print("\nInformation Extracted:")
    print("  JSON Approach:")
    print("    - Name, Company, Department")
    print("    - Fixed schema only")

    print("  BAML Approach:")
    print("    - Name, Current Company, Skills, Experience")
    print("    - Education level, Professional summary")
    print("    - Skill proficiency levels and categories")
    print("    - Years of experience")

    print("\nFlexibility:")
    print("  JSON Approach: Requires manual data structuring")
    print("  BAML Approach: Adapts to any CV format automatically")

    print("\nScalability:")
    print("  JSON Approach: Manual effort per new data source")
    print("  BAML Approach: Automatic processing of diverse inputs")

    print("\nUse Case Fit:")
    print("  JSON Approach: Best for structured, consistent data sources")
    print("  BAML Approach: Best for unstructured, varied data sources")

    # Show specific advantages
    print("\n4. SPECIFIC ADVANTAGES")
    print("======================")

    print("BAML Advantages:")
    print("  ✓ Handles unstructured text input")
    print("  ✓ Extracts nuanced information (experience level, skills)")
    print("  ✓ Adapts to different CV formats")
    print("  ✓ Creates richer knowledge graphs")
    print("  ✓ Scales to new data types without code changes")

    print("\nJSON Advantages:")
    print("  ✓ Faster processing (no LLM calls)")
    print("  ✓ Deterministic results")
    print("  ✓ Lower operational costs")
    print("  ✓ Simpler debugging")
    print("  ✓ No dependency on external LLM services")

    print("\n5. WHEN TO USE EACH APPROACH")
    print("=============================")

    print("Use JSON Approach when:")
    print("  • Data is already structured")
    print("  • Processing speed is critical")
    print("  • LLM costs are a concern")
    print("  • Schema is fixed and simple")
    print("  • Deterministic results required")

    print("\nUse BAML Approach when:")
    print("  • Data is unstructured (CVs, documents, etc.)")
    print("  • Rich information extraction needed")
    print("  • Schema may evolve over time")
    print("  • Intelligence and adaptability valued")
    print("  • Processing diverse data formats")


async def demonstrate_search_differences():
    """
    Demonstrate how the different approaches affect search capabilities.
    """
    print("\n6. SEARCH CAPABILITY COMPARISON")
    print("===============================")

    print("JSON Approach Search Queries:")
    print("  • 'Find people in Engineering department'")
    print("  • 'Who works at TechNova Inc.?'")
    print("  • 'List companies in Technology industry'")
    print("  ↳ Limited to predefined fields")

    print("\nBAML Approach Search Queries:")
    print("  • 'Find people with Python expertise'")
    print("  • 'Who has cloud computing experience?'")
    print("  • 'Find senior engineers with 5+ years experience'")
    print("  • 'Who has machine learning skills?'")
    print("  • 'Find candidates with AWS certification'")
    print("  ↳ Rich semantic search across extracted information")


if __name__ == "__main__":
    asyncio.run(run_comparison())
    asyncio.run(demonstrate_search_differences())
