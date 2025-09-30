# LangChain Tools

This directory contains tools for LangChain agents and workflows.

## SQL Tool Factory

The `sql_tool_factory.py` module provides a factory for creating SQL querying tools that can be used with LangChain agents. These tools combine SQL querying capabilities with language model-based natural language processing.

### Features

- **Generic tool creation**: Create SQL tools for any database with custom configurations
- **Natural language queries**: Convert natural language questions to SQL queries using LLMs
- **Example-driven**: Support for providing example queries to guide the model
- **Configurable**: Customize tool names, descriptions, and behavior
- **Easy integration**: Compatible with existing LangChain agent frameworks

### Usage

#### Basic Usage

```python
from src.ai_core.llm_factory import get_llm
from src.ai_extra.tools_langchain.sql_tool_factory import SQLToolFactory, SQLToolConfig

# Create configuration
config = SQLToolConfig(
    database_uri="sqlite:///path/to/database.db",
    tool_name="my_sql_tool",
    tool_description="Query my database for information",
    examples=[{
        "input": "List all users",
        "query": "SELECT * FROM users LIMIT 10"
    }]
)

# Create tool
factory = SQLToolFactory(get_llm())
tool = factory.create_tool(config)

# Use the tool
result = tool.invoke({"query": "How many users are there?"})
print(result)
```


#### Configuration from Dictionary

```python
from src.ai_extra.tools_langchain.sql_tool_factory import create_sql_tool_from_config

config = {
    "database_uri": "sqlite:///chinook.db",
    "tool_name": "query_chinook",
    "tool_description": "Query the Chinook music database",
    "examples": [
        {
            "input": "List all artists",
            "query": "SELECT * FROM Artist LIMIT 10"
        }
    ]
}

tool = create_sql_tool_from_config(get_llm(), config)
```

### CLI Usage

#### Using ReAct Agent (Recommended)

The recommended way to use SQL tools is through the generic `react-agent` command:

```bash
# Query with ReAct agent (provides conversational interface)
uv run cli react-agent --config "Chinook Music Database" --input "List all artists"

# Use with specific LLM
echo "Which country's customers spent the most?" | uv run cli react-agent --config "Chinook Music Database" --llm gpt_4o_openai

# Interactive chat mode
uv run cli react-agent --config "Chinook Music Database" --chat
```

### Configuration File

SQL tool configurations are defined in `config/demos/react_agent.yaml` as part of the ReAct agent demos:

```yaml
react_agent_demos:
  - name: "Chinook Music Database"
    tools:
      - factory: src.ai_extra.tools_langchain.sql_tool_factory:create_sql_tool_from_config
        config:
          database_uri: "sqlite:///use_case_data/other/Chinook.db"
          tool_name: "query_chinook"
          tool_description: "Query the Chinook music database for information about artists, albums, tracks, customers, and invoices"
          examples:
            - input: "List all artists"
              query: "SELECT * FROM Artist LIMIT 10;"
            - input: "Find all albums for the artist 'AC/DC'"
              query: "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');"
          top_k: 15
    examples:
      - "List all artists in the database"
      - "Which country's customers spent the most?"
      - "Show me albums by AC/DC"
```

### Integration with Existing Code

The SQL tool factory has been integrated into the existing maintenance agent:

```python
# In src/demos/maintenance_agent/tools.py
from src.ai_extra.tools_langchain.sql_tool_factory import SQLToolFactory, SQLToolConfig

# Replace the old hardcoded get_planning_info with:
config = SQLToolConfig(
    database_uri=dummy_database(),
    tool_name="get_planning_info",
    tool_description="Useful for when you need to answer questions about tasks assigned to employees.",
    examples=examples[:5],
)
factory = SQLToolFactory(get_llm())
get_planning_info = factory.create_tool(config)
```

### Architecture

The SQL tool factory leverages:

1. **SQLToolConfig**: Pydantic model for configuration validation
2. **SQLToolFactory**: Main factory class for tool creation
3. **create_sql_querying_graph**: Existing LangGraph-based SQL agent
4. **LangChain @tool decorator**: For creating compatible tools

The factory creates tools that:
1. Accept natural language queries
2. Convert them to SQL using LLMs and examples
3. Execute the SQL against the configured database
4. Return natural language responses

### Error Handling

The factory includes robust error handling for:
- Database connection failures
- Invalid configurations
- SQL execution errors
- LLM API issues

### Compatibility

- Compatible with all LangChain agent frameworks
- Works with any SQL database supported by SQLAlchemy
- Supports various LLM providers through the unified LLM factory