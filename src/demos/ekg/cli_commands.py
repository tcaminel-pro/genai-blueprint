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

LLM_ID = None
KV_STORE_ID = "file"


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def structured_extract(
        file_or_dir: Annotated[
            Path,
            typer.Argument(
                help="Markdown files or directories to process",
                exists=True,
                file_okay=True,
                dir_okay=True,
            ),
        ],
        schema: str = typer.Argument(help="name of he schcme dict to use to extract information"),
        llm_id: Annotated[
            Optional[str], Option("--llm-id", "-m", help="LLM model ID (use list-models to see options)")
        ] = None,
        recursive: bool = typer.Option(False, help="Search for files recursively"),
        batch_size: int = typer.Option(5, help="Number of files to process in each batch"),
        force: bool = typer.Option(False, "--force", help="Overwrite existing KV entries"),
    ) -> None:
        """Extract structured project data from Markdown files and save as JSON in a key-value store.

        Example:
           uv run cli extract-rainbow "*.md" "projects/*.md" --output-dir=./json_output --llm-id gpt-4o
           uv run cli extract-rainbow "**/*.md" --recursive --output-dir=./data
        """

        from loguru import logger

        from src.ai_core.llm_factory import LlmFactory
        from src.demos.ekg.struct_rag_doc_processing import StructuredRagConfig, StructuredRagDocProcessor, get_schema
        from src.utils.pydantic.kv_store import PydanticStore

        schema_dict = get_schema(schema)

        if llm_id is not None and llm_id not in LlmFactory.known_items():
            logger.error(f"Unknown llm_id: {llm_id}. Valid options: {LlmFactory.known_items()}")
            return

        if schema_dict is None:
            logger.error(f"Invalid schema_name: {schema}")
            return
        top_class: str | None = schema_dict.get("top_class")
        if top_class is None:
            logger.error(f"Incorrect schema: {schema}")
            return

        logger.info(f"Starting project extraction with: {file_or_dir} and schema {schema} (class: {top_class})")

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

        vector_store_factory = StructuredRagConfig.get_vector_store_factory()
        struct_rag_conf = StructuredRagConfig(
            model_definition=schema_dict,
            vector_store_factory=vector_store_factory,
            llm_id=None,
            kvstore_id=KV_STORE_ID,
        )
        rag_processor = StructuredRagDocProcessor(rag_conf=struct_rag_conf)
        # Filter out files that already have JSON in KV unless forced
        if not force:
            unprocessed_files = []
            for md_file in md_files:
                key = md_file.stem
                cached_doc = PydanticStore(kvstore_id=KV_STORE_ID, model=struct_rag_conf.get_top_class()).load_object(
                    key
                )
                if not cached_doc:
                    unprocessed_files.append(md_file)
                else:
                    logger.info(f"Skipping {md_file.name} - JSON already exists (use --force to overwrite)")
            md_files = unprocessed_files

        if not md_files:
            logger.info("All files have already been processed. Use --force to reprocess.")
            return
        asyncio.run(rag_processor.process_files(md_files, batch_size))

        logger.success(f"Project extraction complete. {len(md_files)} files processed. Results saved to KV Store")

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

        from src.demos.ekg.struct_rag_tool_factory import create_structured_rag_tool
        from src.utils.cli.langchain_setup import setup_langchain
        from src.utils.cli.langgraph_agent_shell import run_langgraph_agent_shell

        if not setup_langchain(llm_id, lc_debug, lc_verbose, cache):
            return

        rainbow_schema = "Rainbow File"

        rainbow_tool = create_structured_rag_tool(rainbow_schema, llm_id=LLM_ID, kvstore_id=KV_STORE_ID)
        asyncio.run(run_langgraph_agent_shell(llm_id, tools=[rainbow_tool], mcp_server_names=mcp))
