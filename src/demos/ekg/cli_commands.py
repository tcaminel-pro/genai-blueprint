import asyncio
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field
from typer import Option
from upath import UPath

from src.ai_core.prompts import dedent_ws, def_prompt


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def extract_rainbow(
        file_or_dir: Annotated[
            Path,
            typer.Argument(
                help="Markdown files or directories to process",
                exists=True,
                file_okay=True,
                dir_okay=True,
            ),
        ],
        output_dir: Annotated[
            Path,
            typer.Argument(
                help="Output directory",
                exists=True,
                file_okay=False,
                dir_okay=True,
            ),
        ],
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        recursive: bool = typer.Option(False, help="Search for files recursively"),
        batch_size: int = typer.Option(5, help="Number of files to process in each batch"),
        force: bool = typer.Option(False, "--force", help="Overwrite existing JSON files"),
    ) -> None:
        """Extract structured project data from Markdown files and save as JSON.

        Example:
           uv run cli extract-rainbow "*.md" "projects/*.md" --output-dir=./json_output --llm-id gpt-4o
           uv run cli extract-rainbow "**/*.md" --recursive --output-dir=./data
        """

        from loguru import logger
        from upath import UPath

        from src.ai_core.llm import LlmFactory

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            logger.error(f"Unknown llm_id: {llm_id}. Valid options: {LlmFactory.known_items()}")
            return

        logger.info(f"Starting project extraction with: {file_or_dir}")

        # Collect all Markdown files
        all_files = []

        if file_or_dir.is_file() and file_or_dir.suffix.lower() in [".md", ".markdown"]:
            # Single Markdown file
            all_files.append(file_or_dir)
        elif file_or_dir.is_dir():
            # Directory - find Markdown files inside
            if recursive:
                md_files = list(file_or_dir.rglob("*.[mM][dD]"))  # Case-insensitive match
            else:
                md_files = list(file_or_dir.glob("*.[mM][dD]"))
            all_files.extend(md_files)
        else:
            logger.error(f"Invalid path: {file_or_dir} - must be a Markdown file or directory")
            return

        md_files = all_files  # All files are already Markdown files at this point

        if not md_files:
            logger.warning("No Markdown files found matching the provided patterns.")
            return

        logger.info(f"Found {len(md_files)} Markdown files to process")

        # Filter out files that already have JSON output unless forced
        output_path = UPath(output_dir)
        if not force:
            unprocessed_files = []
            for md_file in md_files:
                json_output_file = output_path / f"{md_file.stem}_extracted.json"
                if not json_output_file.exists():
                    unprocessed_files.append(md_file)
                else:
                    logger.info(f"Skipping {md_file.name} - JSON already exists (use --force to overwrite)")
            md_files = unprocessed_files

        if not md_files:
            logger.info("All files have already been processed. Use --force to reprocess.")
            return

        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")
        asyncio.run(process_markdown_batch(md_files, output_path, batch_size))

        logger.success(f"Project extraction complete. {len(md_files)} files processed. Results saved to {output_dir}")

    @cli_app.command()
    def generate_fake_rainbow(
        input_files: Annotated[
            list[Path], typer.Argument(help="JSON files containing project reviews to use as templates")
        ],
        output_dir: Annotated[
            Path,
            typer.Argument(
                help="Output directory for generated fake JSON files",
                file_okay=False,
                dir_okay=True,
            ),
        ],
        count: Annotated[int, Option("--count", "-n", help="Number of fake projects to generate")] = 10,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
    ) -> None:
        """Generate fake but realistic project review JSON files based on existing ones.

        This command reads existing project review JSON files and uses an LLM to generate
        new similar but fake project data that maintains the structure and realistic patterns
        of the originals.

        Example:
           uv run cli generate-fake-projects projects/*.json --output-dir=./fake_data --count=20
           uv run cli generate-fake-projects sample_project.json --output-dir=./generated --count=5
        """
        from src.ai_core.llm import LlmFactory

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            logger.error(f"Unknown llm_id: {llm_id}. Valid options: {LlmFactory.known_items()}")
            return

        # Validate input files exist
        valid_files = []
        for file_path in input_files:
            if not file_path.exists():
                logger.error(f"Input file does not exist: {file_path}")
                return
            if not file_path.suffix.lower() == ".json":
                logger.error(f"Input file must be JSON: {file_path}")
                return
            valid_files.append(file_path)

        if not valid_files:
            logger.error("No valid JSON input files provided")
            return

        output_path = UPath(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {count} fake project reviews based on {len(valid_files)} template files")
        asyncio.run(generate_fake_projects_async(valid_files, output_path, count, llm_id))
        logger.success(f"Generated {count} fake project reviews in {output_dir}")


async def process_markdown_batch(md_files: list[UPath], output_dir: UPath, batch_size: int = 5) -> None:
    """Process a batch of markdown files using LangChain batching."""

    from src.ai_core.llm import get_llm
    from src.ai_core.prompts import def_prompt
    from src.demos.ekg.rainbow_model import RainbowProjectAnalysis

    # Setup LLM with structured output
    llm = get_llm(temperature=0.0).with_structured_output(RainbowProjectAnalysis)

    system = """
    Extract structured information from a project review file. Analyse the files to extracts a comprehensive 360° snapshot of a project: who is involved (team, customer, partners),
    what is being delivered (scope, objectives, technologies), when (start/end dates), where (locations, business lines), how        
    (bidding strategy, pricing, risk mitigation), why (success metrics, differentiators), and how much (TCV, revenue, margin). 

    Details to be extracted is in JSON schema. Answer with a JSON document. Always fill all required fields, possibly with empty list, dict or str.
    """

    user = """
    project review file: 
    ---
    {file}
    ---
    """

    chain = def_prompt(system=system, user=user) | llm

    # Prepare inputs for batch processing
    file_contents = []
    valid_files = []

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            file_contents.append({"file": content})
            valid_files.append(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    if not file_contents:
        logger.warning("No valid files to process")
        return

    # Process in batches
    logger.info(f"Processing {len(valid_files)} files in batches of {batch_size}")

    for i in range(0, len(file_contents), batch_size):
        batch_files = valid_files[i : i + batch_size]
        batch_inputs = file_contents[i : i + batch_size]

        try:
            # Process batch
            results = await chain.abatch(batch_inputs)

            # Save results
            for file_path, result in zip(batch_files, results, strict=False):
                output_file = output_dir / f"{file_path.stem}_extracted.json"
                output_file.write_text(result.model_dump_json(indent=2))
                logger.info(f"Saved: {output_file}")

        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
            # Process files individually on batch failure
            for file_path, input_data in zip(batch_files, batch_inputs, strict=False):
                try:
                    result = await chain.ainvoke(input_data)
                    output_file = output_dir / f"{file_path.stem}_extracted.json"
                    output_file.write_text(result.model_dump_json(indent=2))
                    logger.info(f"Saved (individual): {output_file}")
                except Exception as e2:
                    logger.error(f"Error processing {file_path}: {e2}")


class FakeProjectRequest(BaseModel):
    """Request model for generating a fake project."""

    template_projects: list[dict] = Field(description="List of template project data")
    index: int = Field(description="Index of the fake project being generated")


@tool
def save_fake_project_file(project_data: dict, filename: str) -> str:
    """Save generated fake project data to a JSON file.

    Args:
        project_data: The fake project data to save
        filename: Name of the file to save (without extension)

    Returns:
        Success message with file path
    """
    from upath import UPath

    output_file = UPath(filename).with_suffix(".json")
    output_file.write_text(json.dumps(project_data, indent=2))
    return f"Successfully saved fake project to {output_file}"


async def generate_fake_projects_async(
    input_files: list[Path], output_dir: UPath, count: int, llm_id: str | None = None
) -> None:
    """Generate fake project data based on existing JSON files."""
    from src.ai_core.llm import get_llm

    # Load template projects from input files
    template_projects = []
    for file_path in input_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
                # Handle both single project and array formats
                if isinstance(data, list):
                    template_projects.extend(data)
                else:
                    template_projects.append(data)
        except Exception as e:
            logger.error(f"Error reading template file {file_path}: {e}")
            return

    if not template_projects:
        logger.error("No valid project data found in input files")
        return

    logger.info(f"Loaded {len(template_projects)} template projects")

    # Setup LLM with tools
    llm = get_llm(llm_id=llm_id, temperature=0.7).bind_tools([save_fake_project_file])

    system = dedent_ws( """
        You are an expert at generating realistic fake project data based on templates.
        Your task is to create fake but realistic project review data that follows the same patterns 
        and structure as the provided templates, but with completely fictional information.
        Guidelines:
        - Maintain the exact JSON structure and field names from the templates
        - Generate realistic but fake data (different company names, technologies, dates, etc.)
        - Ensure dates are realistic and consistent
        - Vary the data significantly from the templates while keeping it believable
        - Fill all required fields with appropriate fake data
        - Use diverse company names, project names, and technologies
        - Make sure financial figures are realistic for the project type
        After generating each fake project, use the save_fake_project_file tool to save it.""")

    human = dedent_ws("""
                      Generate fake project #{index} based on these templates:\n
                        {templates_json}

                     Create a unique, realistic fake project with different details but similar structure.""")          

    chain = def_prompt(system, human)  | llm

    # Generate fake projects
    for i in range(count):
        templates_json = json.dumps(template_projects, indent=2)

        # replace code hereafter by a camm to LangGraph create_react_agent AI!
        try:
            response = await chain.ainvoke({"templates_json": templates_json, "index": i + 1})

            # Check if the tool was called
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call.get("name") == "save_fake_project_file":
                        args = tool_call.get("args", {})
                        if "project_data" in args and "filename" in args:
                            filename = output_dir / f"{args['filename']}.json"
                            filename.write_text(json.dumps(args["project_data"], indent=2))
                            logger.info(f"Saved fake project: {filename}")
                            continue

            # Fallback: extract JSON from response and save manually
            import re

            content = str(response)
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    fake_project = json.loads(json_match.group())
                    filename = output_dir / f"fake_project_{i + 1}.json"
                    filename.write_text(json.dumps(fake_project, indent=2))
                    logger.info(f"Saved fake project (manual fallback): {filename}")
                except Exception as e:
                    logger.error(f"Failed to parse/save fake project {i + 1}: {e}")

        except Exception as e:
            logger.error(f"Error generating fake project {i + 1}: {e}")
            # Try individual generation as fallback
            try:
                filename = output_dir / f"fake_project_{i + 1}.json"
                fake_project = await generate_single_fake_project(llm, template_projects, i + 1)
                filename.write_text(json.dumps(fake_project, indent=2))
                logger.info(f"Saved fake project (fallback): {filename}")
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {e2}")


async def generate_single_fake_project(llm, template_projects: list[dict], index: int) -> dict:
    """Fallback method to generate a single fake project without tools."""
    from src.ai_core.prompts import def_prompt

    templates_json = json.dumps(template_projects, indent=2)

    system = """Generate a single realistic fake project review based on the provided templates.
    
    Rules:
    - Use the exact same JSON structure as the templates
    - Create completely fictional but realistic data
    - Vary names, dates, technologies, and financial figures
    - Ensure all required fields are filled
    - Return only valid JSON"""

    user = f"Create fake project #{index} based on these templates:\n\n{templates_json}"

    chain = def_prompt(system=system, user=user) | llm

    # Try to get JSON response
    response = await chain.ainvoke({})
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    # Try to parse JSON from response
    import re

    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    # If parsing fails, create a basic fake project
    template = template_projects[0] if template_projects else {}
    fake_project = {}
    for key, value in template.items():
        if isinstance(value, str):
            fake_project[key] = f"fake_{key}_{index}"
        elif isinstance(value, (int, float)):
            fake_project[key] = value + index
        elif isinstance(value, list):
            fake_project[key] = [f"fake_item_{i}" for i in range(min(len(value), 3))]
        else:
            fake_project[key] = value

    return fake_project
