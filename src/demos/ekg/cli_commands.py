"""CLI commands for EKG (Enterprise Knowledge Graph) document processing and analysis.

This module provides command-line interface commands for:
- Extracting structured data from Markdown documents using LLMs
- Generating synthetic/fake project data based on existing templates
- Running interactive agents for querying the knowledge graph

The commands integrate with the PydanticRAG system to provide document analysis,
vector storage, and semantic search capabilities.

Key Features:
    - Batch processing of Markdown files with parallel LLM extraction
    - Generation of realistic synthetic project data for testing
    - Interactive ReAct agents for querying processed project information
    - Integration with MCP (Model Context Protocol) servers for extended capabilities
    - Configurable caching strategies and LLM selection

Usage Examples:
    ```bash
    # Extract structured data from Markdown files
    uv run cli rainbow-extract "*.md" --output-dir ./data

    # Generate fake project data from existing JSON templates
    uv run cli rainbow-generate-fake "templates/*.json" --output-dir ./fake --count 5

    # Start interactive agent for querying the knowledge graph
    uv run cli ekg-agent-shell --llm-id gpt-4o-mini --mcp filesystem

    # Process recursively with custom settings
    uv run cli rainbow-extract ./reviews/ --recursive --batch-size 10 --force

    # Debug mode for troubleshooting
    uv run cli ekg-agent-shell --debug --verbose --cache sqlite
    ```

Data Flow:
    1. Markdown files → rainbow_extract → JSON structured data
    2. JSON templates → rainbow_generate_fake → Synthetic JSON data
    3. Processed data → ekg_agent_shell → Interactive querying via ReAct agent
"""

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from typer import Option
from upath import UPath

from src.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def rainbow_extract(
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

        from src.ai_core.llm_factory import LlmFactory

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
    def rainbow_generate_fake(
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
        from src.ai_core.llm_factory import LlmFactory
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

    @cli_app.command()
    def ekg_agent_shell(
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        mcp: Annotated[
            list[str], typer.Option(help="MCP server names to connect to (e.g. playwright, filesystem, ..)")
        ] = [],
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
    ) -> None:
        """Run a ReAct agent to query the Enterprise Knowledge Graph (EKG).

        Starts an interactive shell with a ReAct agent that can query processed project
        data using semantic search across the vector store. The agent has access to
        project information extracted from Markdown files and can answer complex queries
        about projects, teams, timelines, and relationships.

        The agent integrates with MCP (Model Context Protocol) servers for extended
        capabilities like file system access, web browsing, or other external tools.

        Examples:
            ```bash
            # Basic interactive shell
            uv run cli ekg-agent-shell

            # With custom LLM and cache
            uv run cli ekg-agent-shell --llm-id gpt-4o-mini --cache sqlite

            # With MCP servers for extended capabilities
            uv run cli ekg-agent-shell --mcp filesystem --mcp playwright

            # Debug mode for troubleshooting
            uv run cli ekg-agent-shell --debug --verbose
            ```

        Interactive Usage:
            Once started, you can ask questions like:
            - "Find all projects using Python"
            - "Which projects had budgets over $1M?"
            - "List team members who worked on healthcare projects"
            - "Compare project delivery times across different technologies"
        """

        from demos.ekg.retriever_tool_factory import PydanticRag
        from src.utils.cli.langchain_setup import setup_langchain
        from src.utils.cli.langgraph_agent_shell import run_langgraph_agent_shell

        if not setup_langchain(llm_id, lc_debug, lc_verbose, cache):
            return

        list_demos = (
            global_config().merge_with("config/demos/document_extractor.yaml").get_list("Document_extractor_demo")
        )
        vector_store_factory = PydanticRag.get_vector_store_factory()
        rainbow_schema = next((item for item in list_demos if item.get("schema_name") == "Rainbow File"))
        rag = PydanticRag(
            model_definition=rainbow_schema,
            vector_store_factory=vector_store_factory,
            llm_id=None,
            kv_store_id="file",
        )
        rainbow_tool = rag.create_vector_search_tool()
        asyncio.run(run_langgraph_agent_shell(llm_id, tools=[rainbow_tool], mcp_server_names=mcp))


async def process_markdown_batch(md_files: list[UPath], output_dir: UPath, batch_size: int = 5) -> None:
    """Process a batch of markdown files using LangChain batching.

    Efficiently processes multiple Markdown files in parallel using LangChain's batch
    processing capabilities. Extracts structured project data and saves results as
    JSON files with error handling and fallback to individual processing.

    Args:
        md_files: List of Markdown file paths to process
        output_dir: Directory where JSON output files will be saved
        batch_size: Number of files to process in each parallel batch

    Processing Pipeline:
        1. Reads all Markdown files from input paths
        2. Processes files in parallel batches using configured LLM
        3. Extracts structured data into RainbowProjectAnalysis format
        4. Saves results as JSON with "_extracted.json" suffix
        5. Handles batch failures by retrying individual files

    Error Handling:
        - Skips unreadable files with error logging
        - Continues processing on batch failures (retries individually)
        - Provides detailed logging for troubleshooting
        - Preserves partial results even if some files fail

    Example:
        ```python
        # Process 15 files in batches of 5
        await process_markdown_batch(
            [Path("review1.md"), Path("review2.md"), ...],
            Path("./output"),
            batch_size=5
        )
        ```
    """

    from src.ai_core.llm_factory import get_llm
    from src.ai_core.prompts import def_prompt

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

    # Prepare document IDs and contents
    document_ids = []
    markdown_contents = []
    valid_files = []

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            document_ids.append(file_path.stem)
            markdown_contents.append(content)
            valid_files.append(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    if not document_ids:
        logger.warning("No valid files to process")
        return

    # Process all documents using batch analysis
    logger.info(f"Processing {len(valid_files)} files in batches of {batch_size}")
    analyzed_docs = await rag.abatch_analyze_documents(document_ids, markdown_contents)

    # Save results
    for doc, file_path in zip(analyzed_docs, valid_files):
        if doc:
            output_file = output_dir / f"{file_path.stem}_extracted.json"
            output_file.write_text(doc.model_dump_json(indent=2))
            logger.info(f"Saved: {output_file}")
