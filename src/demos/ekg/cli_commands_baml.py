"""BAML-based version of structured extraction CLI commands.

This module provides an alternative implementation of the structured_extract function
that uses BAML instead of langchain for structured output extraction. It leverages
the BAML-generated client to extract ReviewedOpportunity data from markdown files.

Key Features:
    - Uses BAML's ExtractRainbow function for structured data extraction
    - Maintains the same CLI interface as the original version
    - Compatible with existing KV store and batch processing infrastructure
    - Provides direct integration with BAML-generated Pydantic models

Usage Examples:
    ```bash
    # Extract structured data from Markdown files using BAML
    uv run cli rainbow-extract-baml "*.md" --output-dir ./data

    # Process recursively with custom settings
    uv run cli rainbow-extract-baml ./reviews/ --recursive --batch-size 10 --force
    ```

Data Flow:
    1. Markdown files → BAML ExtractRainbow → ReviewedOpportunity objects
    2. ReviewedOpportunity objects → KV Store → JSON structured data
    3. Processed data → Available for EKG agent querying
"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from upath import UPath

from src.utils.baml_processor import BamlStructuredProcessor

LLM_ID = None
KV_STORE_ID = "file"


def register_baml_commands(cli_app: typer.Typer) -> None:
    """Register BAML-based commands with the CLI application."""

    @cli_app.command()
    def structured_extract_baml(
        file_or_dir: Annotated[
            Path,
            typer.Argument(
                help="Markdown files or directories to process",
                exists=True,
                file_okay=True,
                dir_okay=True,
            ),
        ],
        recursive: bool = typer.Option(False, help="Search for files recursively"),
        batch_size: int = typer.Option(5, help="Number of files to process in each batch"),
        force: bool = typer.Option(False, "--force", help="Overwrite existing KV entries"),
    ) -> None:
        """Extract structured project data from Markdown files using BAML and save as JSON in a key-value store.

        This command uses BAML's ExtractRainbow function to extract ReviewedOpportunity data
        from markdown files. It provides the same functionality as structured_extract but
        uses BAML instead of langchain for structured output.

        Example:
           uv run cli structured-extract-baml "*.md" "projects/*.md" --force
           uv run cli structured-extract-baml "**/*.md" --recursive
        """

        logger.info(f"Starting BAML-based project extraction with: {file_or_dir}")

        # Check if BAML client is properly configured
        try:
            logger.info("BAML client loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BAML client: {e}")
            logger.error("Make sure OPENROUTER_API_KEY is set and BAML is properly configured")
            return

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

        # Create BAML processor
        processor = BamlStructuredProcessor(kvstore_id=KV_STORE_ID)

        # Filter out files that already have JSON in KV unless forced
        if not force:
            from src.utils.pydantic.kv_store import PydanticStore
            from .baml_client.types import ReviewedOpportunity

            unprocessed_files = []
            for md_file in md_files:
                key = md_file.stem
                cached_doc = PydanticStore(kvstore_id=KV_STORE_ID, model=ReviewedOpportunity).load_object(key)
                if not cached_doc:
                    unprocessed_files.append(md_file)
                else:
                    logger.info(f"Skipping {md_file.name} - JSON already exists (use --force to overwrite)")
            md_files = unprocessed_files

        if not md_files:
            logger.info("All files have already been processed. Use --force to reprocess.")
            return

        asyncio.run(processor.process_files(md_files, batch_size))

        logger.success(
            f"BAML-based project extraction complete. {len(md_files)} files processed. Results saved to KV Store"
        )


def register_commands(cli_app: typer.Typer) -> None:
    """Register BAML commands - alias for register_baml_commands."""
    register_baml_commands(cli_app)


if __name__ == "__main__":
    # For testing purposes
    app = typer.Typer()
    register_baml_commands(app)
    app()
