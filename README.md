# GenAI Blueprint

This project has several goals:
- Support GenAI training, to illustrate key concepts, programming patterns and techniques
- Act as a repository of well integrated reusable components 
- Act as a template for cloud-deployable, highly configurable AI Agents 
- Propose ready to use agentic GenAI demos, demonstrating notably:
  - ReAct Agents to connect different data sources
  - Tool calling agent calling an API
  - Semantic and Hybrid search
  - Researcher agents
  - SmolAgents
  - CrewAI

It's based mainly on the LangChain ecosystem, and integrate many other nice solutions.

## Install
* Use 'poetry' to install and manage project
  * Run 'poetry update' 
  * Note : you can avoid calling "poetry run' by installing the 'shell' command: poetry self add poetry-plugin-shell

* Application settings are in file : app_conf.yaml ; Should likely be edited (and improved...)

* Run 'poetry run python python/main_cli.py echo "hello"  ' to check CLI
* Run 'poetry run python python/main_cli.py run joke  for a quick end-to-end test. add '--help' to see the different options
* Run 'make test' 
  - There some issues with several tests in //. 
  - You might need to change section 'pytest' app_conf.yaml too



### Key Files and Directories

#### Configuration
- `app_conf.yaml`: Main configuration file for LLMs, embeddings, vector stores and chains
- `models_providers.yaml`: Contains model definitions and provider configurations

#### Core AI Components facilitating LangChain programming
- `python/ai_core/`: Core AI infrastructure
  - `llm.py`: LLM factory and configuration
  - `embeddings.py`: Embeddings factory and management
  - `vector_store.py`: Vector store factory and operations
  - `chain_registry.py`: Runnable component registry
  - `cache.py`: LLM caching implementation

#### Reusable AI Components
- `python/ai_extra/`: Generic and reusable AI components, integrated with LangChain
  - `react_agent_structured_output.py` : A ReAct agent generating Pydantic output
  - `gpt_researcher_chain.py` : a LCEL encapsulation og GPT Researcher (A researcher agent)
  - `smolagants_streamlit.py` : Display SmolAgents execution trace in Streamlit UI


#### Demos with UI
- `python/GenAI_Lab.py`: Main Streamlit web application
- `python/pages/`: Streamlit page implementations
  - `1_▫️_Runnable Playground.py`: Page to test registered LangChain runnables
  - `2_▫️_MaintenanceAgent.py`: a ReAct agent to help maintenance planning
  - `3_▫️_Stock_Price.py`: a tool calling agent to ger and compare stock prices
  - `4_▫️_DataFrame.py`: a tool calling agent query tabular data
  - `5_▫️_Mon_Master.py`: Example of similarity search project
  - `7_▫️_GPT_Researcher.py`: Page demonstrating GPT Researcher
  - `9_▫️_SmallAgents` : Page to execute SmollAgents
  - `12_▫️_Crew_AI.py`: CrewAI demonstration
  - ...

#### Notebooks for GenAI training 
- `python/Notebooks`

#### Examples (more for training purpose) 
- `python/ai_chains/`: Example chains and RAG implementations. Examples: 
  - `B_1_naive_rag_example.py`: Basic RAG implementation
  - `B_2_self_query.py`: Self-querying retriever example
  - `C_1_tools_example.py`: Tool usage demonstration
  - `C_2_advanced_rag_langgraph.py`: Advanced RAG with LangGraph
  - `C_3_essay_writer_agent.py`: Essay writing agent
  - ...

#### Utilities
- `python/utils/`: Utility functions and helpers
  - `streamlit/`: Streamlit-specific utilities
  - `pydantic/`: Pydantic model extensions
  - `singleton.py`: Singleton pattern implementation

#### Testing and Development
- `tests/`: Unit and integration tests
- `Makefile`: Common development tasks
- `pyproject.toml`: Poetry project configuration
- `Dockerfile`: optimized dockerfile (for Azure)
- `CONVENTION.md`: Coding convention used by Aider-chat (a coding assistant)


