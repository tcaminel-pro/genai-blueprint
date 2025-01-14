# GenAI Training 


## Install
* Use 'poetry' to install and manage project
  * run 'poetry shell'
  * run 'poetry update'  

* Application settings are in file : app_conf.yaml ; Should likely be edited (and improved...)

* Run 'make test' - But there some issues with several tests in //. You might need to change section 'pytest' app_conf.yaml too
* Run 'python python/main_cli.py echo "hello"  ' to check CLI
* Run 'python python/main_cli.py run joke  for a quick end-to-end test. add '--help' to see the different options

* Run 'make webapp' to launch the Web application server. URL is http://localhost:8501/ 
* Run 'make fastapi' to launch FastAPI


### Key Files and Directories

#### Configuration
- `app_conf.yaml`: Main configuration file for LLMs, embeddings, vector stores and chains
- `models_providers.yaml`: Contains model definitions and provider configurations

#### Core AI Components
- `python/ai_core/`: Core AI infrastructure
  - `llm.py`: LLM factory and configuration
  - `embeddings.py`: Embeddings factory and management
  - `vector_store.py`: Vector store factory and operations
  - `chain_registry.py`: Runnable component registry
  - `cache.py`: LLM caching implementation


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



#### Demos and Examples
- `python/ai_chains/`: Example chains and RAG implementations. Examples: 
  - `B_1_naive_rag_example.py`: Basic RAG implementation
  - `B_2_self_query.py`: Self-querying retriever example
  - `C_1_tools_example.py`: Tool usage demonstration
  - `C_2_advanced_rag_langgraph.py`: Advanced RAG with LangGraph
  - `C_3_essay_writer_agent.py`: Essay writing agent
  - ...

#### Web Interface
- `python/GenAI_Lab.py`: Main Streamlit web application
- `python/pages/`: Streamlit page implementations
    `0_▫️_Runnable Playground.py`: Page to test registered LangChain runnables
  - `5_▫️_Mon_Master.py`: Example of similarity search project
  - `12_▫️_Crew_AI.py`: CrewAI demonstration
  - ...
