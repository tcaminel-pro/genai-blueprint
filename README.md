# GenAI Framework

A production-ready framework for building and deploying Generative AI applications with:

- **Core Components**: Factories for LLMs, Embeddings, Vector Stores and Runnables
- **Modular Architecture**: Plug-and-play components for AI workflows
- **Agent Systems**:
  - ReAct and Plan-and-Execute agents
  - Multi-tool calling agents
  - Hybrid search (semantic + keyword)
  - Research and data analysis agents
  - Multi-agent coordination (CrewAI, MCP, AutoGen)

Built on LangChain with extensions for enterprise use cases.

## Core Stack

**Foundation**:
- `LangChain` - AI orchestration
- `LangGraph` - Agent workflows  
- `Pydantic` - Data validation
- `FastAPI` - REST endpoints
- `Streamlit` - Web interfaces
- `Typer` - CLI framework

**Key Integrations**:
- `LlamaIndex` - Advanced RAG
- `Tavily` - Web search
- `GPT Researcher` - Autonomous research
- `MCP` - Model coordination
- `AutoGen` - Multi-agent systems
- `Weaviate` - Vector database

## Documentation

For an overview of the code structure and patterns:
[Tutorial: genai-blueprint](https://code2tutorial.com/tutorial/d4f58807-1657-41e1-92b8-15a3a10cb162/index.md) 

Note: The tutorial is automatocally generated and may be slightly outdated - refer to the code for current implementations.

## Getting Started

**Prerequisites**:
- Python 12 (installed automatically via `uv`)
- `make` and `uv` for dependency management

**Installation**:
```bash
make install
```

**Configuration**:
- Main settings: `config/app_conf.yaml`
- API keys: `.env` file in project root or parent directories

**Quick Test**:
```bash
make test_install  # Verifies basic functionality
make test         # Runs test suite (some parallel tests may need adjustment)
```

Configure LLMs via `/config/providers.yaml` after setting up API keys.



### Key Files and Directories

#### Configuration
- `config/baseline.yaml`: Main configuration file for LLMs, embeddings, vector stores and chains.  
- `config/models_providers.yaml`: Contains model definitions and provider configurations
- `pyproject.toml`: uv project configuration


#### Core AI Components facilitating LangChain programming
- `src/ai_core/`: Core AI infrastructure
  - `llm.py`: LLM factory and configuration
  - `embeddings.py`: Embeddings factory and management
  - `vector_store.py`: Vector store factory and operations
  - `chain_registry.py`: Runnable component registry
  - `cache.py`: LLM caching implementation
  - `vision.py`: Facilitate use of multimodal LLM
  - `prompts.py`: Prompt templates and utilities
  - `structured_output.py`: Structured output generation helper
  - `instrumentation.py`: Telemetry and monitoring

#### Reusable AI Components
- `src/ai_extra/`: Generic and reusable AI components, integrated with LangChain
  - `gpt_researcher_chain.py`: LCEL encapsulation of GPT Researcher
  - `smolagents_chain.py`: SmolAgents implementation
  - `mcp_tools.py`: Model Context Protocol utilities


#### Demos with UI
- `src/webapp/`: Streamlit web application and pages
  - `pages/`: Streamlit page implementations
  - `1_▫️_Runnable Playground.py`: Page to test registered LangChain runnables
  - `2_▫️_MaintenanceAgent.py`: a ReAct agent to help maintenance planning
  - `3_▫️_Stock_Price.py`: a tool calling agent to ger and compare stock prices
  - `4_▫️_DataFrame.py`: a tool calling agent query tabular data
  - `5_▫️_Mon_Master.py`: Example of similarity search project
  - `7_▫️_GPT_Researcher.py`: Page demonstrating GPT Researcher
  - `9_▫️_SmallAgents` : Page to execute SmollAgents
  - `12_▫️_Crew_AI.py`: CrewAI demonstration
  - ...

#### Utilities
- `src/utils/`: Utility functions and helpers
  - `config_mngr.py`: Configuration management
  - `pydantic/`: Pydantic model extensions
  - `streamlit/`: Streamlit-specific utilities
  - `singleton.py`: Singleton pattern implementation
  - `units.py`: Unit conversion utilities


#### Notebooks for GenAI training 
- `python/Notebooks`

#### Reusable UI Components
- `src/webapp/ui_components/`: Generic and reusable User Interface components for Streamlit
  - `llm_config.py`: Form to select LLM and common configuration parameters
  - `capturing_callback_handler.py`: Callback handler for Streamlit
  - `clear_result.py`: State management utilities

#### Examples (more for training purpose)
- `src/ai_chains/`: Example chains and RAG implementations. Examples:
  - `B_1_naive_rag_example.py`: Basic RAG implementation
  - `B_2_self_query.py`: Self-querying retriever example
  - `C_1_tools_example.py`: Tool usage demonstration
  - `C_2_advanced_rag_langgraph.py`: Advanced RAG with LangGraph
  - `C_3_essay_writer_agent.py`: Essay writing agent
  - ...


#### Testing and Development
- `tests/`: Unit and integration tests
- `src/demos/`: Various demonstration implementations
  - `maintenance_agent/`: Maintenance planning demo
  - `mon_master_search/`: Search demo
  - `todo/`: Task management demos
- `Makefile`: Common development tasks
- `Dockerfile`: optimized dockerfile 
- `CONVENTION.md`: Coding convention used by Aider-chat (a coding assistant)


## CLI Usage Examples

**Core Operations**:
```bash
# LLM queries
uv run llm "Explain quantum computing" --model gpt-4 --stream
cat document.txt | uv run llm --task summarize

# Chain execution  
uv run chain research --input "AI safety trends" --verbose
uv run chain analyze --file data.csv --output report.md

# RAG pipelines
uv run rag index --dir ./docs --collection research_papers
uv run rag query "neural networks" --top-k 5

# Agent Systems
uv run agent research "latest AI papers" --tools web-search,arxiv
uv run agent analyze --file financial_report.pdf --agent financial-analyst
```

**Advanced Features**:
```bash
# Multi-agent workflows
uv run multiagent project-plan --agents planner,researcher,editor

# Data processing
uv run process pdf-to-text *.pdf --output ./text_files
uv run process extract-tables document.pdf --format csv

# System Management
uv config set llm.default_model=gpt-4
uv config show --all
uv tools list --category web-search
```
