import os
from pathlib import Path

from pydantic import BaseModel
from src.ai_core.llm import get_llm
from src.ai_core.prompts import dedent_ws, def_prompt
from src.demos.ekg.rainbow_model import RainbowProjectAnalysis

from langgraph.prebuilt import create_react_agent


class RainbowProjectAnalysisList(BaseModel)
    array: list[RainbowProjectAnalysis]


def generate_fake_rainbows(examples : list[Path], number_of_generated_fakes: int, output_dir: Path)
    LLM_ID = None
    llm = get_llm(temperature=0.0).with_structured_output(RainbowProjectAnalysisList)

    system = dedent_ws("""
        You are an expert at generating 10 realistic fake projects data based on templates.
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
        """)

    human = dedent_ws("""
            Generate fake project #{index} based on these templates:\n
            {templates_json}

            Create a unique, realistic fake project with different details but similar structure.""")

def test() -> None:
    file1 = Path(os.getenv("ONEDRIVE","")) / "prj/atos-kg/rainbow-json/03.RESM-SOL-9000559500_CNES_TMA_VENUS_VIP_PEPS_THEIA_MUSCATE-v0.2_extracted.json"
    assert file1.exists()
    generate_fake_rainbows([file1], 3, Path("/temp"))

if __name__ == "__main__":
    test()