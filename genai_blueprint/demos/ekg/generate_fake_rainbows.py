"""
Generate fake project Rainbow review data based on real project templates.

This module provides functionality to create realistic but fake project Rainbow review data using LangChain.
It loads JSON templates of actual project reviews and generates new synthetic data with similar structure
but completely fictional content including company names, technologies, team members, and financial data.
"""

import json
import os
from pathlib import Path

from genai_blueprint.demos.ekg.rainbow_model import RainbowProjectAnalysis
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.prompts import dedent_ws, def_prompt
from loguru import logger
from pydantic import BaseModel


class RainbowProjectAnalysisList(BaseModel):
    """A container for multiple RainbowProjectAnalysis instances."""

    array: list[RainbowProjectAnalysis]


def generate_fake_rainbows_from_samples(
    examples: list[Path], number_of_generated_fakes: int, output_dir: Path, llm_id: str | None = None
) -> None:
    """
    Generate fake project reviews based on templates and save as JSON files.

    Args:
        examples: List of paths to JSON template files containing real project reviews
        number_of_generated_fakes: Number of fake projects to generate
        output_dir: Directory where generated JSON files will be saved
        llm: LangChain LLM instance to use for generation. If None, uses default from get_llm()
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    all_examples = [example.read_text() for example in examples]
    logger.info(f"Loaded {len(all_examples)} templates, generating {number_of_generated_fakes} fake projects")

    system = dedent_ws("""
        You are an expert at generating realistic fake project data based on examples.
        Your task is to create fake but realistic project review data that follows the same patterns 
        and structure as the provided examples, but with completely fictional information.
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

    user = dedent_ws("""
        Generate {count} unique, realistic fake projects based on these templates:\n
        {templates_json}\n
        Create diverse fake projects with different details but similar structure and complexity.""")

    llm = get_llm(llm_id, temperature=0.7).with_structured_output(RainbowProjectAnalysisList)
    chain = def_prompt(system, user) | llm
    templates_json = "\n".join(all_examples)
    try:
        response = chain.invoke({"count": number_of_generated_fakes, "templates_json": templates_json})

        # Save each fake project as a separate JSON file
        for fake_project in response.array:
            # Use identification.name and identification.opportunity-id for filename
            name = fake_project.identification.name
            opportunity_id = fake_project.identification.opportunity_id
            safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).rstrip()
            safe_name = safe_name.replace(" ", "_")
            output_file = output_dir / f"{safe_name}_{opportunity_id}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(fake_project.model_dump(), f, indent=2, default=str)
            logger.info(f"Saved fake project {name}_{opportunity_id} to {output_file}")

    except Exception as e:
        logger.error(f"Error generating fake projects: {e}")
        raise


def test() -> None:
    """Test the fake rainbow generation with sample files."""
    file1 = (
        Path(os.getenv("ONEDRIVE", ""))
        / "prj/atos-kg/rainbow-json/03.RESM-SOL-9000559500_CNES_TMA_VENUS_VIP_PEPS_THEIA_MUSCATE-v0.2_extracted.json"
    )
    assert file1.exists()

    output_dir = Path("/tmp")
    generate_fake_rainbows_from_samples([file1], 2, output_dir, llm_id=None)
    logger.info(f"Test completed. Check {output_dir} for generated files.")


if __name__ == "__main__":
    test()
