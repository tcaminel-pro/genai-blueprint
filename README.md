# GenAI Blueprint

This project has several goals:
- Support GenAI training, to illustrate key concepts, programming patterns and techniques
- Implement the `Factory pattern` for LLM, Embeddings models, Vector Store, Runnables
- Act as a repository of well integrated reusable components 
- Act as a template for cloud-deployable, highly configurable AI Agents 
- Propose ready to use agentic GenAI demos, demonstrating notably:
  - ReAct Agents to connect different data sources
  - Tool calling agent calling an API
  - Semantic and Hybrid search
  - Researcher agents
  - MCP
  - CrewAI

It's based mainly on the LangChain ecosystem, and integrate many other nice solutions.

The core stack includes:
  - `LangChain`, `LangGraph`, `LangServe`, ...
  - `OmegaConf`: configuration management
  - `Streamlit`: User Interface (web)
  - `Typer`: Command Line Interface
  - `FastAPI`: REST APIs
  - `Pydantic` for ..  a lot

Extra components uses:
  - `SmallAgents` : Agent Framework
  - `GPR Researcher` : Deep Internet Search
  - `MCP Adapt` : Model Contect Protocol client
  - ...

## Install
We use make and uv. This command install uv if not present, then load the project dependencies.
* `make install` 


Configuration:
* Application settings are in file : `app_conf.yaml` ; Should likely be edited (and improved...)
* API keys are taken from  a `.env` file, in the project directory or its parents 

Quick test:
* Run `echo "computers" | uv run src/main_cli.py run joke  -m fake_parrot_local` 
  * It should display 'Tell me a joke on computers' 
  * Don't care about the warnings
  * add `--help` to see the different options
  * You can change the LLM by taking one in defined in `models_providers.yaml`  (if keys are in the `.env` file)
* Run 'make test' 
  - There some issues with several tests in //. 
  - You might need to change section 'pytest' `app_conf.yaml` too



### Key Files and Directories

#### Configuration
- `app_conf.yaml`: Main configuration file for LLMs, embeddings, vector stores and chains.  
- `models_providers.yaml`: Contains model definitions and provider configurations
- `pyproject.toml`: Poetry project configuration


#### Core AI Components facilitating LangChain programming
- `src/ai_core/`: Core AI infrastructure
  - `llm.py`: LLM factory and configuration
  - `embeddings.py`: Embeddings factory and management
  - `vector_store.py`: Vector store factory and operations
  - `chain_registry.py`: Runnable component registry
  - `cache.py`: LLM caching implementation
  - `vision.py`: Facilitate use of multimodal LLM
  - `prompts.py`: Prompt templates and utilities
  - `structured_output.py`: Structured output generation
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


