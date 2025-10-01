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
    uv run cli ekg-agent-shell --llm gpt-4o-mini --mcp filesystem

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
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from typer import Option
from upath import UPath

LLM_ID = None
KV_STORE_ID = "file"


async def call_ekg_agent(
    query: str, llm_id: str | None = None, tools: list | None = None, mcp_server_names: list[str] | None = None
) -> None:
    """Execute a single query using the EKG agent with tools and stream the response.

    Creates a ReAct agent with EKG tools and optional MCP tools, then streams the
    response to the query. This is similar to call_react_agent but for EKG agents.

    Args:
        query: The input query to process
        llm_id: Optional ID of the language model to use
        tools: List of tools (e.g., structured RAG tools) to include in the agent
        mcp_server_names: Optional list of MCP server names to include

    Example:
        await call_ekg_agent(
            "Find all projects using Python",
            llm_id="gpt-4o-mini",
            tools=[rainbow_tool],
            mcp_server_names=["filesystem"]
        )
    """
    from genai_tk.core.llm_factory import get_llm_unified
    from genai_tk.core.mcp_client import get_mcp_servers_dict
    from genai_tk.utils.langgraph import print_astream
    from langchain_core.messages import HumanMessage
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
    from loguru import logger

    model = get_llm_unified(llm=llm_id)
    all_tools = tools or []

    # Add MCP tools if specified
    if mcp_server_names:
        logger.info(f"Connecting to MCP servers: {mcp_server_names}")
        client = MultiServerMCPClient(get_mcp_servers_dict(mcp_server_names))
        mcp_tools = await client.get_tools()
        all_tools.extend(mcp_tools)

    agent = create_react_agent(model, all_tools)
    logger.info("Executing EKG agent query...")
    resp = agent.astream({"messages": [HumanMessage(content=query)]})
    await print_astream(resp)


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
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
        recursive: bool = typer.Option(False, help="Search for files recursively"),
        batch_size: int = typer.Option(5, help="Number of files to process in each batch"),
        force: bool = typer.Option(False, "--force", help="Overwrite existing KV entries"),
    ) -> None:
        """Extract structured project data from Markdown files and save as JSON in a key-value store.

        Example:
           uv run cli extract-rainbow "*.md" "projects/*.md" --output-dir=./json_output --llm-id gpt-4o
           uv run cli extract-rainbow "**/*.md" --recursive --output-dir=./data
        """

        from genai_tk.core.llm_factory import LlmFactory
        from genai_tk.utils.pydantic.kv_store import PydanticStore
        from loguru import logger

        from genai_blueprint.demos.ekg.struct_rag_doc_processing import (
            StructuredRagConfig,
            StructuredRagDocProcessor,
            get_schema,
        )

        # Resolve LLM identifier if provided
        llm_id = None
        if llm:
            resolved_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm)
            if error_msg:
                print(error_msg)
                return
            llm_id = resolved_id

        schema_dict = get_schema(schema)

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
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
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
        from genai_tk.core.llm_factory import LlmFactory

        from genai_blueprint.demos.ekg.generate_fake_rainbows import generate_fake_rainbows_from_samples

        # Resolve LLM identifier if provided
        llm_id = None
        if llm:
            resolved_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm)
            if error_msg:
                print(error_msg)
                return
            llm_id = resolved_id

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
        input: Annotated[str | None, typer.Option(help="Input query or '-' to read from stdin")] = None,
        cache: Annotated[str, typer.Option(help="Cache strategy: 'sqlite', 'memory' or 'no_cache'")] = "memory",
        mcp: Annotated[
            list[str], typer.Option(help="MCP server names to connect to (e.g. playwright, filesystem, ..)")
        ] = [],
        lc_verbose: Annotated[bool, Option("--verbose", "-v", help="Enable LangChain verbose mode")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d", help="Enable LangChain debug mode")] = False,
        llm: Annotated[Optional[str], Option("--llm", "-m", help="LLM identifier (ID or tag from config)")] = None,
        chat: Annotated[bool, Option("--chat", help="Start an interactive shell to send prompts")] = False,
    ) -> None:
        """Run a ReAct agent to query the Enterprise Knowledge Graph (EKG).

        Can either start an interactive shell or execute a single query directly.
        The agent can query processed project data using semantic search across the
        vector store and has access to project information extracted from Markdown files.

        The agent integrates with MCP (Model Context Protocol) servers for extended
        capabilities like file system access, web browsing, or other external tools.

        Examples:
            ```bash
            # Basic interactive shell
            uv run cli ekg-agent-shell

            # Execute a single query
            uv run cli ekg-agent-shell --input "Find all projects using Python"

            # Read query from stdin
            echo "Which projects had budgets over $1M?" | uv run cli ekg-agent-shell

            # With custom LLM and cache
            uv run cli ekg-agent-shell --llm gpt-4o-mini --cache sqlite

            # With MCP servers for extended capabilities
            uv run cli ekg-agent-shell --mcp filesystem --mcp playwright --input "List files and analyze project data"

            # Force interactive shell even with input
            uv run cli ekg-agent-shell --shell --input "This will be ignored"

            # Debug mode for troubleshooting
            uv run cli ekg-agent-shell --debug --verbose
            ```

        Usage modes:
            - No --input parameter: Start interactive shell (default behavior)
            - With --input: Execute single query and exit
            - With --chat: Force interactive shell mode (ignores --input)
            - Reading from stdin: Pipe input directly to the agent

        Interactive Usage:
            Once started, you can ask questions like:
            - "Find all projects using Python"
            - "Which projects had budgets over $1M?"
            - "List team members who worked on healthcare projects"
            - "Compare project delivery times across different technologies"
        """

        from genai_tk.core.llm_factory import LlmFactory
        from genai_tk.utils.cli.langchain_setup import setup_langchain
        from genai_tk.utils.cli.langgraph_agent_shell import run_langgraph_agent_shell

        from genai_blueprint.demos.ekg.struct_rag_tool_factory import create_structured_rag_tool

        # Resolve LLM identifier if provided
        llm_id = None
        if llm:
            resolved_id, error_msg = LlmFactory.resolve_llm_identifier_safe(llm)
            if error_msg:
                print(error_msg)
                return
            llm_id = resolved_id

        if not setup_langchain(llm_id, lc_debug, lc_verbose, cache):
            return

        rainbow_schema = "Rainbow File"
        rainbow_tool = create_structured_rag_tool(rainbow_schema, llm_id=LLM_ID, kvstore_id=KV_STORE_ID)

        if chat:
            # Force interactive shell mode (ignores input parameter)
            asyncio.run(run_langgraph_agent_shell(llm_id, tools=[rainbow_tool], mcp_server_names=mcp))
        else:
            # Handle input parameter or stdin
            if not input and not sys.stdin.isatty():
                input = sys.stdin.read()

            if input and len(input.strip()) >= 5:
                # Execute single query and exit
                asyncio.run(call_ekg_agent(input.strip(), llm_id=llm_id, tools=[rainbow_tool], mcp_server_names=mcp))
            else:
                # No input provided or input too short, start interactive shell (default behavior)
                if input and len(input.strip()) < 5:
                    print("Warning: Input too short (minimum 5 characters), starting interactive shell instead")
                asyncio.run(run_langgraph_agent_shell(llm_id, tools=[rainbow_tool], mcp_server_names=mcp))
