# GenAI Framework

A production-ready framework for building and deploying Generative AI applications with the following features:

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
- `Pydantic` - Data modelisation & validation
- `FastAPI` - REST endpoints
- `Streamlit` - Web interfaces
- `Typer` - CLI framework
- `OmegaConf` - Configuration Management


**Key Integrations**:
- `Tavily` - Web search
- `GPT Researcher` - Autonomous research
- `MCP` - Model Context Protocol
- `AutoGen` - Multi-agent systems
- `pgvector` - Vector database

## Documentation

For an overview of the code structure and patterns:
[Tutorial: genai-blueprint](https://code2tutorial.com/tutorial/d4f58807-1657-41e1-92b8-15a3a10cb162/index.md) 

Note: The tutorial is automatically generated and may be slightly outdated - refer to the code for current implementations.

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
- `config/baseline.yaml`: Main configuration file for LLMs, embeddings, vector stores and chains.  
- `config/overide.yaml`: overriden configuration. Can be selected by an environment variable  
- `config/providers/llm.yaml` and `embeddings.yaml`: Contains model definitions and provider configurations
- `config/mcp_servers.yaml` : MCP Servers parameters 
- `pyproject.toml`: uv project configuration
- `config/demos`: Demos related configuration


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
  - `llm.py`: LLM factory and configuration
  - `embeddings.py`: Embeddings factory and management
  - `vector_store.py`: Vector store factory and operations
  - `chain_registry.py`: Runnable component registry
  - `cache.py`: LLM caching implementation
  - `langgraph_runner.py`: LangGraph session management
  - `prompts.py`: Prompt templates and utilities
  - `bm25s_retriever.py`: BM25 retriever implementation
  - `react_agent_structured_output.py`: ReAct agent with structured output

#### Reusable AI Components
- `src/ai_extra/`: Generic and reusable AI components
  - `bm25s_retriever.py`: Fast BM25 retriever without Elasticsearch
  - `react_agent_structured_output.py`: ReAct agent with structured output

#### Demos and Examples
- `src/demos/`: Various demonstration implementations
  - `maintenance_agent/`: Maintenance planning demo with dummy data
  - `mon_master_search/`: Hybrid search demo
  - `todo/`: Task management demos
    - `agent.py`: Basic todo agent
    - `human-in-loop-agent.py`: Human-in-the-loop agent

- `src/webapp/`: Streamlit web application
  - `pages/`: Streamlit page implementations
    - `5_mon_master.py`: Hybrid search UI
    - `7_deep_search_agent.py`: Research agent with logging
    - `10_codeAct_agent.py`: CodeAct agent implementation
    - `12_reAct_agent.py`: ReAct agent implementation  
    - `14_graph_RAG.py`: Graph-based RAG demo
    - `18_residio_anonymization.py`: Anonymization demo
  - `ui_components` : reusable streamlit components
    -`smolagents_streamlit.py`: SmolAgents UI components
    - `streamlit_chat.py` : helper to display LangGraph chat in streamlit


#### Utilities
- `src/utils/`: Utility functions and helpers
  - `config_mngr.py`: Configuration management with OmegaConf
  - `basic_auth.py`: Basic authentication utilities
  - `crew_ai/remove_telemetry.py`: CrewAI telemetry removal
  - `langgraph.py`: LangGraph utilities
  - `pydantic/field_adder.py`: Pydantic field manipulation
  - `singleton.py`: Singleton pattern implementation
  - `streamlit/`: Streamlit-specific utilities
    - `clear_result.py`: State management
    - `load_data.py`: Data loading utilities
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
    - `docker.mk`
    - `aws.mk`



## CLI Usage Examples
The framework provides several CLI commands for interacting with AI components.
They are registered in the configuration file 
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

**Install Node (for MCP)** 
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
nvm install --lts
```
