import os
import json
from pathlib import Path

from pydantic import BaseModel
from loguru import logger
from src.ai_core.llm import get_llm
from src.ai_core.prompts import dedent_ws, def_prompt
from src.demos.ekg.rainbow_model import RainbowProjectAnalysis

from langchain_core.prompts import ChatPromptTemplate


class RainbowProjectAnalysisList(BaseModel):
    array: list[RainbowProjectAnalysis]


def load_templates(template_files: list[Path]) -> list[RainbowProjectAnalysis]:
    """Load project review templates from JSON files."""
    templates = []
    for file_path in template_files:
        if not file_path.exists():
            logger.warning(f"Template file not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both single project and list formats
            if isinstance(data, list):
                for item in data:
                    templates.append(RainbowProjectAnalysis(**item))
            else:
                templates.append(RainbowProjectAnalysis(**data))
                
        except Exception as e:
            logger.error(f"Error loading template {file_path}: {e}")
            continue
    
    return templates


def generate_fake_rainbows(examples: list[Path], number_of_generated_fakes: int, output_dir: Path) -> None:
    """Generate fake project reviews based on templates and save as JSON files."""
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load templates
    templates = load_templates(examples)
    if not templates:
        logger.error("No valid templates loaded. Cannot generate fakes.")
        return
    
    logger.info(f"Loaded {len(templates)} templates, generating {number_of_generated_fakes} fake projects")
    
    # Create LLM with structured output
    llm = get_llm(temperature=0.7).with_structured_output(RainbowProjectAnalysisList)
    
    system = dedent_ws("""
        You are an expert at generating realistic fake project data based on templates.
        Your task is to create fake but realistic project review data that follows the same patterns 
        and structure as the provided templates, but with completely fictional information.
        Guidelines:
        - Generate realistic but fake data (different company names, technologies, dates, etc.)
        - Ensure dates are realistic and consistent
        - Ensure some people are on different projects with similar roles
        - Vary the data significantly from the templates while keeping it believable
        - Fill all required fields with appropriate fake data
        - Use diverse company names, project names, and technologies
        - Make sure financial figures are realistic for the project type
        - Ensure all JSON fields are populated with appropriate data types
        """)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", """Generate {count} unique, realistic fake projects based on these templates:

        {templates_json}
        
        Create diverse fake projects with different details but similar structure and complexity.""")
    ])
    
    # Prepare templates as JSON
    templates_json = json.dumps([t.model_dump() for t in templates], indent=2, default=str)
    
    # Generate fake projects in batches
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({
            "count": number_of_generated_fakes,
            "templates_json": templates_json
        })
        
        # Save each fake project as a separate JSON file
        for i, fake_project in enumerate(response.array):
            output_file = output_dir / f"fake_project_{i+1:03d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(fake_project.model_dump(), f, indent=2, default=str)
            logger.info(f"Saved fake project {i+1} to {output_file}")
            
    except Exception as e:
        logger.error(f"Error generating fake projects: {e}")
        raise


def test() -> None:
    """Test the fake rainbow generation with sample files."""
    # Use a more accessible test path
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test template if no file exists
    test_file = test_dir / "test_template.json"
    if not test_file.exists():
        from src.demos.ekg.rainbow_model import (
            ProjectDetails, TechnicalInfo, FinancialInfo, 
            TeamMember, Stakeholder, Risk, Deliverable
        )
        
        test_project = RainbowProjectAnalysis(
            project_details=ProjectDetails(
                project_name="Test Project",
                company="Test Corp",
                start_date="2023-01-15",
                end_date="2023-12-15",
                project_type="Software Development"
            ),
            technical_info=TechnicalInfo(
                technologies=["Python", "React", "PostgreSQL"],
                complexity="Medium",
                team_size=5
            ),
            financial_info=FinancialInfo(
                budget=100000,
                actual_cost=95000,
                roi=1.2
            ),
            team_members=[
                TeamMember(name="John Doe", role="Tech Lead", email="john@test.com")
            ],
            stakeholders=[
                Stakeholder(name="Jane Smith", role="Project Manager", email="jane@test.com")
            ],
            risks=[
                Risk(description="Timeline risk", impact="Medium", probability=0.3)
            ],
            deliverables=[
                Deliverable(name="Final Report", status="Completed", due_date="2023-12-15")
            ]
        )
        
        with open(test_file, 'w') as f:
            json.dump([test_project.model_dump()], f, indent=2, default=str)
    
    output_dir = test_dir / "fake_output"
    generate_fake_rainbows([test_file], 2, output_dir)
    logger.info(f"Test completed. Check {output_dir} for generated files.")

if __name__ == "__main__":
    test()
