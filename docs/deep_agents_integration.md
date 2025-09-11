# Deep Agents Integration Documentation

## Overview

The Deep Agents integration provides powerful AI agents with planning capabilities, sub-agents, file system access, and real-time web search. The integration follows the `deepagents` package patterns while adding enterprise-ready features.

## Key Components

### 1. Core Integration Module (`src/ai_core/deep_agents.py`)
- **DeepAgentFactory**: Factory class for creating and managing deep agents
- **DeepAgentConfig**: Configuration model for agent settings
- **Specialized Agents**: Pre-configured agents for research, coding, and data analysis
- **Async Support**: Full async/await support for streaming responses

### 2. Search Tools Module (`src/ai_core/search_tools.py`)
- **Multi-Provider Support**: Automatically selects the best available search provider
- **Priority Order**:
  1. Tavily API (best quality, requires API key)
  2. Serper API (Google results, requires API key)
  3. DuckDuckGo (free, no API key required)
  4. Mock search (fallback for testing)
- **Two Interfaces**:
  - `create_search_function()`: Returns raw Python function for deep agents
  - `create_search_tool()`: Returns LangChain tool for compatibility

### 3. CLI Integration (`src/ai_core/cli_commands.py`)
Commands added:
- `cli deep-agent <type>`: Run deep agents from command line
- `cli list-deep-agents`: List all created agents
- `cli deep-agent-demo`: Launch interactive Streamlit demo

### 4. Example Implementations
- **Research Agent** (`src/ai_chains/deep_agents/research_agent_example.py`)
- **Coding Agent** (`src/ai_chains/deep_agents/coding_agent_example.py`)
- **Interactive Demo** (`src/demos/deep_agent_demo.py`)

## Usage Examples

### Command Line Usage

```bash
# Research agent with real web search
uv run cli deep-agent research --input "Latest AI developments" --llm-id gpt_41_openrouter

# Coding agent
uv run cli deep-agent coding --input "Write a Python fibonacci function"

# Data analysis agent
uv run cli deep-agent analysis --files data.csv --input "Analyze this dataset"

# Custom agent with specific instructions
uv run cli deep-agent custom --input "Help me plan a project" --instructions "You are a project manager"
```

### Python Code Usage

```python
from src.ai_core.deep_agents import DeepAgentFactory, run_deep_agent
from src.ai_core.search_tools import create_search_function

# Create factory
factory = DeepAgentFactory()

# Set LLM model
factory.set_default_model("gpt_41_openrouter")

# Create search function (automatically uses best available provider)
internet_search = create_search_function(verbose=True)

# Create research agent
agent = factory.create_research_agent(
    search_tool=internet_search,
    name="My Research Agent",
    async_mode=True
)

# Run the agent
import asyncio

async def main():
    result = await run_deep_agent(
        agent=agent,
        messages=[{"role": "user", "content": "Research quantum computing"}],
        stream=False
    )
    print(result["messages"][-1].content)

asyncio.run(main())
```

## Setting Up Search Providers

### Option 1: Tavily (Recommended)
1. Get free API key at https://tavily.com
2. Set environment variable: `export TAVILY_API_KEY=your-key`
3. Or add to `.env` file

### Option 2: Serper
1. Get API key at https://serper.dev
2. Set environment variable: `export SERPER_API_KEY=your-key`

### Option 3: DuckDuckGo (Free, No API Key)
1. Install package: `pip install duckduckgo-search`
2. Works immediately without configuration

## Architecture

```
┌─────────────────────────────────────┐
│          CLI Interface              │
│     (cli deep-agent commands)       │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│       DeepAgentFactory              │
│   (Creates and manages agents)      │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│        Deep Agent Core              │
│  (Planning, Sub-agents, Tools)      │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│        Search Tools                 │
│  (Tavily/Serper/DuckDuckGo/Mock)   │
└─────────────────────────────────────┘
```

## Key Features

1. **Automatic Search Provider Selection**: The system automatically chooses the best available search provider based on API keys and installed packages.

2. **Streaming Support**: Agents can stream responses for real-time interaction.

3. **File System Access**: Agents can read and write files with proper sandboxing.

4. **Sub-Agent Support**: Complex agents can delegate tasks to specialized sub-agents.

5. **Human-in-the-Loop**: Support for requiring human approval for certain actions.

6. **Async-First Design**: Built for performance with async/await throughout.

## Testing

Run the test suite:
```bash
uv run pytest tests/ai_core/test_deep_agents.py -v
```

Test search functionality:
```bash
python src/ai_core/search_tools.py
```

## Troubleshooting

### Issue: "No real search results"
**Solution**: Set up a search provider (see "Setting Up Search Providers" above)

### Issue: "LLM API key not found"
**Solution**: Configure an LLM model that has API keys set:
```bash
uv run cli config-info  # Check available models
uv run cli deep-agent research --llm-id <available-model> --input "query"
```

### Issue: "Tool binding not supported"
**Solution**: Use an LLM model that supports function calling (most OpenAI, Anthropic, and modern models do)

## Next Steps

1. Add more specialized agents (e.g., DevOps, Database, Security agents)
2. Implement agent memory and conversation history
3. Add agent collaboration features
4. Create agent marketplace for sharing configurations
5. Add monitoring and observability for agent actions

## References

- [Deep Agents Package](https://github.com/LangGraph-Solutions/deepagents)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Tavily API Documentation](https://docs.tavily.com)
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)
