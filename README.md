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
```
Configure LLMs via `/config/providers/llm.yaml` after setting up API keys.



### Key Files and Directories

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