import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from typer import Option
from upath import UPath

from src.ai_core.prompts import def_prompt


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
        file_or_dir: Annotated[
            Path,
            typer.Argument(
                help="JSON files or directories containing project reviews to use as templates",
                exists=True,
                file_okay=True,
                dir_okay=True,
            ),
        ],
        output_dir: Annotated[
            Path,
            typer.Argument(
                help="Output directory for generated fake JSON files",
                file_okay=False,
                dir_okay=True,
            ),
        ],
        count: Annotated[int, Option("--count", "-n", help="Number of fake projects to generate per input file")] = 1,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        recursive: bool = typer.Option(False, help="Search for files recursively"),
    ) -> None:
        """Generate fake but realistic project review JSON files based on existing ones.

        This command reads existing project review JSON files and uses an LLM to generate
        new similar but fake project data that maintains the structure and realistic patterns
        of the originals.

        Example:
           uv run cli generate-fake-rainbow "projects/*.json" --output-dir=./fake_data --count=5
           uv run cli generate-fake-rainbow sample_project.json --output-dir=./generated --count=3
           uv run cli generate-fake-rainbow "data/**/*.json" --recursive --output-dir=./generated
        """
        from src.ai_core.llm import LlmFactory
        from src.demos.ekg.generate_fake_rainbows import generate_fake_rainbows_from_samples

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            logger.error(f"Unknown llm_id: {llm_id}. Valid options: {LlmFactory.known_items()}")
            return

        # Collect all JSON files
        all_files = []

        if file_or_dir.is_file() and file_or_dir.suffix.lower() in [".json"]:
            # Single JSON file
            all_files.append(file_or_dir)
        elif file_or_dir.is_dir():
            # Directory - find JSON files inside
            if recursive:
                json_files = list(file_or_dir.rglob("*.json"))
            else:
                json_files = list(file_or_dir.glob("*.json"))
            all_files.extend(json_files)
        else:
            logger.error(f"Invalid path: {file_or_dir} - must be a JSON file or directory")
            return

        if not all_files:
            logger.warning("No JSON files found matching the provided patterns.")
            return

        logger.info(f"Found {len(all_files)} JSON files to process")

        output_path = UPath(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_generated = 0
        for json_file in all_files:
            logger.info(f"Processing template: {json_file}")
            generate_fake_rainbows_from_samples(
                examples=[json_file], number_of_generated_fakes=count, output_dir=output_path, llm_id=llm_id
            )
            total_generated += count

        logger.success(
            f"Successfully generated {total_generated} fake project reviews from {len(all_files)} templates in {output_dir}"
        )


async def process_markdown_batch(md_files: list[UPath], output_dir: UPath, batch_size: int = 5) -> None:
    """Process a batch of markdown files using LangChain batching."""

    from src.ai_core.llm import get_llm
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
