# GenAI Framework

A framework for building and deploying Generative AI/ Agentic AI applications with the following features:

- **Core Components**: Factories for LLMs, Embeddings, Vector Stores and Runnables
- **Modular Architecture**: Plug-and-play components for AI workflows
- **Agent Systems**:
  - ReAct and Plan-and-Execute agents
  - Multi-tool calling agents
  - Hybrid search (semantic + keyword)
  - Research and data analysis agents
  - Multi-agent coordination (CrewAI, MCP, AutoGen)

- Built on LangChain with extensions for enterprise use cases.
- Extensive use of 'factory' and 'inversion of control' patterns to improve extendibility

## Core Stack

**Foundation**:
- `LangChain` - AI orchestration
- `LangGraph` - Agent workflows  
- `Pydantic` - Data modelisation & validation
- `FastAPI` - REST endpoints
- `Streamlit` - Web interfaces
- `Typer` - CLI framework
- `OmegaConf` - Configuration management


**Key Integrations**:
- `Tavily` - Web search
- `GPT Researcher` - Autonomous research
- `MCP` - Model Context Protocol
- `AutoGen` - Multi-agent systems
- `pgvector` - Vector database

## Documentation

For an overview of the code structure and patterns:
[Tutorial: genai-blueprint](https://code2tutorial.com/tutorial/d4f58807-1657-41e1-92b8-15a3a10cb162/index.md) 

Note: The tutorial was automatically generated and may be slightly outdated - refer to the code for current implementations.

## Getting Started

**Prerequisites**:
- Python 3.12 (installed automatically via `uv`). The code should however work with Python 3.11 or 3.13.
- `uv` for dependency management
- `make` for build commands

It has been tested on Linux, WSL/Ubuntu and MacOS.

**Installation**:
```bash
make install
```

**Configuration**:
- Main settings: `config/app_conf.yaml`
- API keys: `.env` file in project root or parent directories
- `config/baseline.yaml`: Main configuration file for LLMs, embeddings, vector stores and chains
- `config/overrides.yaml`: Override configuration (can be selected by environment variable)
- `config/providers/`: Provider-specific configurations
  - `llm.yaml`: LLM model definitions and provider configurations
  - `embeddings.yaml`: Embedding model configurations
- `config/mcp_servers.yaml`: MCP server parameters
- `config/demos/`: Demo-specific configurations
  - `cli_examples.yaml`, `codeact_agent.yaml`, `react_agent.yaml`, etc.
- `config/components/`: Component-specific configurations
  - `cognee.yaml`, `gpt_researcher.yaml`
- `config/schemas/`: Schema definitions
  - `document_extractor.yaml`
- `pyproject.toml`: uv project configuration


**Quick Test**:
```bash
make test_install  # Verifies basic functionality
make test         # Runs test suite (some parallel tests may need adjustment)
make webapp       # lauch the Streamlit app
```
Configure LLMs via `/config/providers/llm.yaml` after setting up API keys.



### Key Files and Directories

#### Core AI Components
- `src/ai_core/`: Core AI infrastructure
  - `llm_factory.py`: LLM factory and configuration
  - `embeddings_factory.py`: Embeddings factory and management
  - `vector_store_factory.py`: Vector store factory and operations
  - `chain_registry.py`: Runnable component registry
  - `cache.py`: LLM caching implementation
  - `langgraph_runner.py`: LangGraph session management
  - `prompts.py`: Prompt templates and utilities
  - `deep_agents.py`: Deep learning agent implementations
  - `mcp_client.py`: Model Context Protocol client
  - `structured_output.py`: Structured output handling
  - `providers.py`: Provider configuration and management
  - `cli_commands.py`: Core CLI command implementations

#### Reusable AI Components
- `src/ai_extra/`: Generic and reusable AI components
  - `autogen_utils.py`: AutoGen model utilities and integrations
  - `browser_use_langchain.py`: Browser automation with LangChain
  - `cognee_utils.py`: Cognee knowledge graph utilities
  - `custom_presidio_anonymizer.py`: Custom anonymization utilities
  - `image_analysis.py`: Image analysis capabilities
  - `kv_store_factory.py`: Key-value store factory
  - `pgvector_factory.py`: PostgreSQL vector store factory
  - `chains/`: Chain implementations
    - `fabric_chain.py`: Fabric pattern integration and content fetching
    - `gpt_researcher_chain.py`: GPT Researcher integration for autonomous research
  - `graphs/`: Graph-based agent implementations
    - `custom_react_agent.py`: Custom ReAct agent using Functional API
    - `react_agent_structured_output.py`: ReAct agent with structured output
    - `sql_agent.py`: SQL querying agent with database integration
  - `loaders/`: Data loading utilities
    - `mistral_ocr.py`: OCR functionality using Mistral API with batch processing
  - `retrievers/`: Information retrieval components
    - `bm25s_retriever.py`: Fast BM25 retriever without Elasticsearch
  - `tools_langchain/`: LangChain-compatible tools
    - `config_loader.py`: Configuration loading tools
    - `multi_search_tools.py`: Multi-search functionality
    - `shared_config_loader.py`: Shared configuration utilities
    - `web_search_tool.py`: Web search integration
  - `tools_smolagents/`: SmolAgents-compatible tools
    - `browser_use.py`: Browser automation for SmolAgents
    - `config_loader.py`: Configuration tools for SmolAgents
    - `dataframe_tools.py`: DataFrame manipulation tools
    - `deep_config_loader.py`: Deep configuration loading
    - `react_config_loader.py`: ReAct agent configuration
    - `sql_tools.py`: SQL tools for SmolAgents
    - `yfinance_tools.py`: Yahoo Finance integration
  - `cli_commands.py` and `cli_commands_agents.py`: Additional CLI commands

#### Demos and Examples
- `src/demos/`: Various demonstration implementations
  - `deep_agents/`: Deep learning agent demonstrations
    - `coding_agent_example.py`: Coding agent implementation
    - `research_agent_example.py`: Research agent demonstration
  - `ekg/`: Enterprise Knowledge Graph demos
    - `cli_commands.py` and `cli_commands_baml.py`: EKG CLI commands
    - `generate_fake_rainbows.py`: Synthetic data generation
    - `struct_rag_doc_processing.py`: Structured RAG document processing
    - `struct_rag_tool_factory.py`: RAG tool factory
    - `test_baml_extract.py`: BAML extraction testing
  - `maintenance_agent/`: System maintenance agent demos
    - `dummy_data.py`: Test data generation
    - `tools.py`: Maintenance tools implementation
  - `mon_master_search/`: Master search functionality
    - `loader.py`: Data loading utilities
    - `model_subset.py`: Model subset management
    - `search.py`: Search implementation

- `src/webapp/`: Streamlit web application
  - `pages/`: Streamlit page implementations organized by category
    - `demos/`: Main demo pages
      - `mon_master.py`: Hybrid search UI
      - `deep_search_agent.py`: Research agent with logging
      - `codeAct_agent.py`: CodeAct agent implementation
      - `reAct_agent.py`: ReAct agent implementation
      - `graph_RAG.py`: Graph-based RAG demo
      - `anonymization.py`: Presidio anonymization demo
      - `cognee_KG.py`: Cognee knowledge graph demo
      - `deep_agent.py`: Deep agent implementations
      - `maintenance_agent.py`: System maintenance agent
    - `settings/`: Configuration and setup pages
      - `welcome.py`: Welcome and overview page
      - `configuration.py`: System configuration interface
      - `MCP_servers.py`: MCP server management
    - `training/`: Educational and training pages
      - `runnable_playground.py`: Interactive runnable testing
      - `tokenization.py`: Tokenization demonstration
      - `CLI_runner.py`: CLI command testing interface
  - `ui_components/`: Reusable Streamlit components
    - `smolagents_streamlit.py`: SmolAgents UI components
    - `streamlit_chat.py`: Helper to display LangGraph chat in Streamlit
    - `config_editor.py`: Configuration editing interface
    - `cypher_graph_display.py`: Graph visualization component
    - `llm_selector.py`: LLM selection component
  - `cli_commands.py`: Webapp-specific CLI commands


#### Utilities
- `src/utils/`: Utility functions and helpers
  - `config_mngr.py`: Configuration management with OmegaConf
  - `basic_auth.py`: Basic authentication utilities
  - `collection_helpers.py`: Collection manipulation utilities
  - `langgraph.py`: LangGraph utilities
  - `load_data.py`: Data loading utilities
  - `logger_factory.py`: Logging configuration factory
  - `markdown.py`: Markdown processing utilities
  - `singleton.py`: Singleton pattern implementation
  - `spacy_model_mngr.py`: spaCy model management
  - `sql_utils.py`: SQL utilities
  - `cli/`: CLI-specific utilities
    - `langchain_setup.py`: LangChain setup utilities
    - `langgraph_agent_shell.py`: LangGraph interactive shell
    - `smolagents_shell.py`: SmolAgents interactive shell
  - `crew_ai/remove_telemetry.py`: CrewAI telemetry removal
  - `pydantic/`: Pydantic utilities
    - `dyn_model_factory.py`: Dynamic model factory
    - `field_adder.py`: Pydantic field manipulation
    - `jsonl_store.py`: JSONL storage utilities
    - `kv_store.py`: Key-value store implementation
  - `streamlit/`: Streamlit-specific utilities
    - `auto_scroll.py`: Auto-scrolling functionality
    - `capturing_callback_handler.py`: Callback handler for capturing
    - `clear_result.py`: State management
    - `recorder.py`: Streamlit action recording
    - `thread_issue_fix.py`: Streamlit threading fixes

#### Testing and Development
- `tests/`: Unit and integration tests
- `src/wip/`: Work in progress
- `Makefile`: Common development and deployment tasks
- `CONVENTION.md`: Coding convention used by Aider-chat (a coding assistant)

#### Deployment
  - `Dockerfile`: Optimized dockerfile
  - `deploy/`: Deployment scripts and configurations
    - `docker.mk` : build and run a container locally
    - `aws.mk` : deploy in AWS
    - `azure.mk` : deploy in Azure
    - `modal.mk` : deploy in Modal


## Streamlit Demos Configuration
- The `Streamit` app can be somewhat configured in `app_conf.yaml`  (key: `ui`).
- Most Demos can be configured with YAML file in `config/demos`


## CLI Usage Examples
- The framework provides several CLI commands, typically for interacting with AI components. 
- They are implemented in `cli_command.py` files, and registered in `app_conf.yaml`  (key: `commands/modules`)


```bash
uv run cli --help   # list of defined commands
```

**Basic LLM Interaction**
```bash
uv run cli llm --input "Hello world"  # Simple LLM query
echo "Hello world" | uv run cli llm  # Pipe input
uv run cli llm --llm-id gpt-4 --stream  # Use specific model with streaming
uv run cli run joke --input "bears"  # Run a joke chain
```

**Agent with tools / MCP**
```bash
uv run cli mcp-agent --server filesystem --shell # start interactive shell
echo "get news from atos.net web site" | uv run cli mcp-agent --server playwright --server filesystem # ReAct Agent
uv run cli smolagents "How many seconds would it take for a leopard at full speed to run through Pont des Arts?" -t web_search  # CodeAct Agent
```

**Deep Agents (Enhanced with beautiful markdown rendering)**
```bash
# Research agent with markdown output
uv run cli deep-agent research --input "Latest AI developments" --llm-id gpt-4

# Coding agent for development tasks
uv run cli deep-agent coding --input "Write a Python async web scraper" --llm-id gpt-4

# Analysis agent for data insights
uv run cli deep-agent analysis --input "Analyze sales trends" --files sales_data.csv --llm-id gpt-4

# Custom agent with specific instructions
uv run cli deep-agent custom --input "Plan a project timeline" --instructions "You are a project manager" --llm-id gpt-4
```

**Misc**
```bash
echo "artificial intelligence" | uv run cli fabric -p "create_aphorisms" --llm-id llama-70-groq # Fabric patterns
uv run cli ocr-pdf "*.pdf" "data/*.pdf" --output-dir=./ocr_results # OCR with Mistral API
```

**Utilities**
```bash
uv run cli list-models  # List available models
uv run cli config-info  # Show current configuration
uv run cli list-mcp-tools --filter playwright  # List available MCP tools
```

## Aditional install (for some demos / components)
**install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma3:4b   # example LLM
ollama pull snowflake-arctic-embed:22m  # example embeddings
```
**Install Chrome and Playwright**
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
uv add playwright
playwright install --with-deps
```

**Spacy models in uv**
```bash
uv pip install pip
uv run --with spacy spacy download fr_core_news_sm
uv run --with spacy spacy download en_core_web_lg 
```
or 
```bash
make install_spacy_models
```

**Install Node (for some MCP servers)** 
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
nvm install --lts
```
