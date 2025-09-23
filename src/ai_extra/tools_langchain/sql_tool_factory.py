"""SQL Tool Factory for LangChain Integration.

This module provides a factory for creating SQL query tools that can be used
with LangChain agents. The factory creates tools that combine SQL querying
capabilities with language model-based natural language processing.
"""

from typing import Any

from langchain.tools import BaseTool, tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from src.ai_extra.graphs.sql_agent import create_sql_querying_graph


class SQLToolConfig(BaseModel):
    """Configuration for SQL tool factory."""

    database_uri: str = Field(description="Database connection URI")
    tool_name: str = Field(default="sql_query", description="Name of the generated tool")
    tool_description: str = Field(
        default="Useful for answering questions by querying the database",
        description="Description of what the tool does",
    )
    examples: list[dict[str, str]] = Field(
        default_factory=list, description="List of example queries with 'input' and 'query' keys"
    )
    top_k: int = Field(default=10, description="Maximum number of results to return")


class SQLToolFactory:
    """Factory for creating SQL querying tools for LangChain agents.

    This factory creates tools that can execute natural language queries
    against SQL databases using a language model to generate and interpret
    SQL queries.

    Example:
    ```
        config = SQLToolConfig(
            database_uri="sqlite:///path/to/database.db",
            tool_name="planning_info",
            tool_description="Query task assignments and schedules",
            examples=[{
                "input": "Tasks for employee X",
                "query": "SELECT * FROM tasks WHERE employee = 'X'"
            }]
        )

        factory = SQLToolFactory(llm)
        tool = factory.create_tool(config)
    ```
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the factory with a language model.

        Args:
            llm: Language model for query generation and answering
        """
        self.llm = llm

    def create_tool(self, config: SQLToolConfig) -> BaseTool:
        """Create a SQL querying tool based on the provided configuration.

        Args:
            config: Configuration specifying database connection and tool behavior

        Returns:
            Configured LangChain tool for SQL querying

        Raises:
            ValueError: If database URI is invalid or database is inaccessible
            ConnectionError: If unable to connect to the database
        """
        # Validate database connection
        try:
            db = SQLDatabase.from_uri(config.database_uri)
            # Test connection by getting table info
            db.get_table_info()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}") from e

        # Create the SQL querying graph
        graph = create_sql_querying_graph(llm=self.llm, db=db, examples=config.examples, top_k=config.top_k)

        @tool
        def sql_query_tool(query: str) -> str:
            """Execute a natural language query against the database."""
            result = graph.invoke({"question": query})
            return result["answer"]

        # Set the tool name and description after creation
        sql_query_tool.name = config.tool_name
        sql_query_tool.description = config.tool_description

        return sql_query_tool

    def create_tool_from_dict(self, config_dict: dict[str, Any]) -> BaseTool:
        """Create a SQL querying tool from a dictionary configuration.

        Args:
            config_dict: Dictionary containing tool configuration

        Returns:
            Configured LangChain tool for SQL querying
        """
        config = SQLToolConfig(**config_dict)
        return self.create_tool(config)



def create_sql_tool_from_config(config: dict[str, Any], llm: BaseChatModel | None = None) -> BaseTool:
    """Create a SQL tool from a configuration dictionary.

    This function provides a simple interface for creating SQL tools
    from configuration files or dictionaries.

    Args:
        config: Configuration dictionary with tool settings
        llm: Language model for query generation and answering (optional, will use default if not provided)

    Returns:
        Configured SQL querying tool

    Example:
    ```
        config = {
            "database_uri": "sqlite:///path/to/db.sqlite",
            "tool_name": "query_chinook",
            "tool_description": "Query the Chinook music database",
            "examples": [
                {
                    "input": "List all artists",
                    "query": "SELECT * FROM Artist LIMIT 10"
                }
            ]
        }

        tool = create_sql_tool_from_config(config)
    ```
    """
    # Get LLM if not provided
    if llm is None:
        from src.ai_core.llm_factory import get_llm

        llm = get_llm()

    factory = SQLToolFactory(llm)
    return factory.create_tool_from_dict(config)
