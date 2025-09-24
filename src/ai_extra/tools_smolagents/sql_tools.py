"""SQL tools for SmolAgents integration.

This module provides SQL database tools for use with SmolAgents,
allowing agents to execute SQL queries on configured database connections.
Refactored to leverage LangChain's SQLDatabase utility for better interoperability.
"""

from typing import Any

import pandas as pd
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel
from smolagents import Tool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.ai_core.prompts import dedent_ws
from src.utils.sql_utils import check_dsn_update_driver


class TableConfig(BaseModel):
    """Configuration for a database table."""

    name: str
    description: str | None = None


class SQLTool(Tool):
    """A tool for executing SQL queries on databases.

    This tool provides SQL query execution capabilities for AI agents,
    connecting to databases using DSN configuration and leveraging LangChain's
    SQLDatabase utility for enhanced functionality and better interoperability.
    """

    name: str
    description: str
    inputs = {"query": {"type": "string", "description": "SQL query to execute"}}
    output_type = "object"
    dsn: str
    tables: list[TableConfig]
    engine: Engine | None = None
    db: SQLDatabase | None = None

    def __init__(self, name: str, dsn: str, description: str, tables: dict[str, Any] | list[str] | None = None) -> None:
        """Initialize SQL tool with database connection and table configuration.

        Args:
            dsn: Database connection string
            description: Description of the database purpose
            tables: Table configuration (dict with name/description or list of names)
            name: Tool name
        """
        super().__init__()
        self.name = name
        self.dsn = dsn
        self.tables = self._parse_tables_config(tables)
        self._initialize_database()
        self.description = self._build_description(description)

    def _parse_tables_config(self, tables: dict[str, Any] | list[str] | None) -> list[TableConfig]:
        """Parse tables configuration from various formats.

        Args:
            tables: Table configuration in various formats

        Returns:
            List of TableConfig objects
        """
        if tables is None:
            return []

        if isinstance(tables, list):
            # List of table names
            return [TableConfig(name=table) for table in tables]
        elif isinstance(tables, dict):
            # Single table with name and description
            if "name" in tables:
                return [TableConfig(**tables)]
            else:
                # Dict of table_name -> config
                return [
                    TableConfig(name=name, **config)
                    if isinstance(config, dict)
                    else TableConfig(name=name, description=str(config))
                    for name, config in tables.items()
                ]
        else:
            return []

    def _build_description(self, database_description: str) -> str:
        """Build comprehensive tool description including schema info.

        Args:
            database_description: Base description of the database

        Returns:
            Enhanced description with schema information
        """
        base_desc = dedent_ws(
            f"""
            This tool executes SQL queries on a database. 
            IMPORTANT:  output is a Thetupple with a Pandas Dataframe and its 50 first rows. \n
            The database is described as:  {database_description}"""
        )

        if not self.tables:
            return base_desc

        schema_info = self._get_detailed_schema_info()
        if schema_info:
            return f"{base_desc}\n\n{schema_info}"
        else:
            return base_desc

    def _initialize_database(self) -> None:
        """Initialize SQLAlchemy engine and LangChain SQLDatabase using DSN."""
        try:
            dsn = check_dsn_update_driver(self.dsn, None)  # Async not supported
            self.engine = create_engine(dsn)

            # Initialize LangChain SQLDatabase with the engine
            self.db = SQLDatabase(engine=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database using DSN '{self.dsn}': {e}") from e

    def _get_detailed_schema_info(self) -> str:
        """Get detailed schema information for configured tables using LangChain SQLDatabase.

        Returns:
            Formatted schema information suitable for LLM consumption
        """
        if not self.db:
            return ""

        try:
            # If specific tables are configured, get schema for those
            if self.tables:
                table_names = [table.name for table in self.tables]
                try:
                    # Use LangChain's get_table_info method for specific tables
                    schema_info = self.db.get_table_info(table_names=table_names)

                    # Enhance with user-provided descriptions
                    enhanced_parts = ["Available tables:"]

                    for table_config in self.tables:
                        table_name = table_config.name
                        enhanced_parts.append(f"\n- Table '{table_name}':")

                        if table_config.description:
                            enhanced_parts.append(f"\n  Description: {table_config.description}")

                        # Add guidance based on table analysis
                        guidance = self._generate_table_guidance(table_name)
                        if guidance:
                            enhanced_parts.append(f"\n  Query guidance: {guidance}")

                    # Add the raw schema information from LangChain
                    enhanced_parts.append(f"\n\nDetailed Schema Information:\n{schema_info}")

                    return "\n".join(enhanced_parts)

                except Exception:
                    # Fallback to all table info if specific tables fail
                    return f"Schema information (fallback): {self.db.get_table_info()}"
            else:
                # No specific tables configured, return all available tables
                return f"Available database schema:\n{self.db.get_table_info()}"

        except Exception as e:
            return f"Schema information not available: {e}"

    def _generate_table_guidance(self, table_name: str) -> str:
        """Generate query guidance for a table based on LangChain schema information.

        Args:
            table_name: Name of the table

        Returns:
            Generated guidance text
        """
        if not self.db:
            return ""

        try:
            # Get table info from LangChain's SQLDatabase
            table_info = self.db.get_table_info(table_names=[table_name])
            table_info_lower = table_info.lower()

            guidance = []

            # Analyze column names in the schema to provide guidance
            if "embedding" in table_info_lower:
                guidance.append("supports vector similarity searches")
            if "tsv" in table_info_lower or "tsvector" in table_info_lower:
                guidance.append("supports full-text search queries")
            if "_id" in table_info_lower or table_info_lower.count("id") > 1:
                guidance.append("use ID fields for filtering and joins")
            if "content" in table_info_lower:
                guidance.append("contains text content for analysis")
            if "metadata" in table_info_lower:
                guidance.append("includes metadata for filtering")

            return ", ".join(guidance) if guidance else "general purpose table"

        except Exception:
            return "general purpose table"

    def forward(self, query: str) -> tuple[pd.DataFrame, str]:  # type: ignore
        """Execute SQL query and return results as DataFrame using LangChain SQLDatabase.

        Args:
            query: SQL query string to execute

        Returns:
            Tuple of (DataFrame containing query results, string representation of first 50 rows)
        """
        if not query or not query.strip():
            raise ValueError("SQL query cannot be empty")

        if not self.db:
            self._initialize_database()

        try:
            # Also get results as DataFrame for compatibility with existing interface
            with self.engine.connect() as conn:  # type: ignore
                result_df = pd.read_sql_query(text(query), conn)

            # Return both DataFrame and string representation
            return (result_df, str(result_df.head(50)))

        except Exception as e:
            raise RuntimeError(f"Failed to execute SQL query: {e}") from e

    def get_schema_info(self) -> str:
        """Get database schema information for context using LangChain SQLDatabase.

        Returns:
            String representation of configured tables and their schemas
        """
        return self._get_detailed_schema_info() or "No schema information available"

    @property
    def langchain_db(self) -> SQLDatabase | None:
        """Provide access to the underlying LangChain SQLDatabase for advanced usage.

        Returns:
            The LangChain SQLDatabase instance if initialized
        """
        return self.db
