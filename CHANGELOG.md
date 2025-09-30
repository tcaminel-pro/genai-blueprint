# Changelog

All notable changes to genai_bp (GenAI Blueprint) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-09-30

### Changed
- **Major Architecture Refactor** - Split monolithic project into two packages:
  - `genai_tk` - Core AI toolkit with reusable components
  - `genai_bp` - Application framework and ready-to-use demos
- **Dependency Management** - Now depends on `genai_tk` from GitHub for all AI capabilities
- **Import Updates** - All imports updated from internal `src.ai_core` → `genai_tk.core` pattern
- **Project Structure** - Reorganized tests into `unit_tests/` and `integration_tests/` directories
- **Build System** - Updated pyproject.toml to reflect new architecture and dependencies

### Added
- **Comprehensive Documentation**
  - New `Agents.md` documenting all Blueprint applications and use cases
  - Updated installation instructions for the split architecture
  - Integration examples and deployment patterns
- **CI/CD Pipeline** - GitHub Actions workflow with genai_tk dependency testing
- **Enhanced Testing** - Cross-package integration validation

### Applications Retained in genai_bp
- **Main Applications** (`src/main/`)
  - CLI interface with AI functionality
  - Streamlit web application
  - FastAPI backend services
  - LangServe deployment configuration
  - Modal serverless deployment

- **Demo Applications** (`src/demos/`)
  - Enterprise Knowledge Graph (EKG) with Kuzu database
  - Deep Agents for research and coding
  - Maintenance Agent for industrial use cases
  - Master Search across document collections

- **Web Interface** (`src/webapp/`)
  - Streamlit pages for training, demos, and settings
  - Interactive UI components for AI interactions
  - Configuration management interfaces
  - Graph visualization components

- **Processing Chains** (`src/ai_chains/`)
  - Basic examples (joke, RAG, self-query)
  - Advanced patterns (LangGraph RAG, agentic retrieval)
  - Essay writer and structured output agents

- **MCP Servers** (`src/mcp_server/`)
  - Math computation server
  - Weather information server
  - Tech news aggregation server

### Migration Guide

#### For Developers
1. **Install genai_bp**: `uv pip install git+https://github.com/tcaminel-pro/genai-blueprint@main`
   - This automatically installs `genai_tk` as a dependency
2. **Import Changes**: Update any custom code using internal imports:
   ```python
   # Old (internal imports)
   from src.ai_core.llm_factory import LLMFactory
   from src.utils.config_mngr import ConfigManager
   
   # New (using genai_tk)
   from genai_tk.core.llm_factory import LLMFactory
   from genai_tk.utils.config_mngr import ConfigManager
   ```

#### For Applications
- **Streamlit**: Run with `make webapp` or `streamlit run src/main/streamlit.py`
- **CLI Tools**: Use `uv run cli --help` for available commands
- **FastAPI**: Deploy with `uvicorn src.main.fastapi_app:app`
- **Modal**: Deploy with `modal deploy src/main/modal_app.py`

### Key Use Cases in genai_bp
1. **Enterprise Document Processing** - Transform documents into queryable knowledge graphs
2. **Industrial Maintenance** - AI-powered maintenance assistance and troubleshooting
3. **Research and Analysis** - Automated research workflows with citation tracking
4. **Code Analysis** - Software development assistance and code generation
5. **Data Visualization** - Interactive data exploration and analysis

### Technical Details
- **Architecture**: Application framework built on genai_tk
- **Dependencies**: Simplified dependencies with AI capabilities from genai_tk
- **Python Version**: Requires Python >=3.12,<3.13
- **Package Manager**: Built for uv with streamlined dependency groups
- **Testing**: Cross-package integration tests with genai_tk

### Deployment Options
- **Development**: `uv sync && make webapp`
- **Streamlit Cloud**: Direct deployment with git URL
- **Docker**: Production-ready containerization
- **Modal**: Serverless deployment for scalable applications
- **FastAPI**: REST API services with Uvicorn

### Breaking Changes
- **Import Paths**: All internal AI component imports must be updated to use genai_tk
- **Dependencies**: Projects depending on internal AI components must now install genai_tk
- **Test Structure**: Tests moved from `tests/` to `unit_tests/` and `integration_tests/`

### Future Roadmap
- Enhanced integration examples
- More demo applications showcasing genai_tk capabilities
- Production deployment templates
- Performance optimization guides