"""SQL tools for SmolAgents integration.

This module provides SQL database tools for use with SmolAgents,
allowing agents to execute SQL queries on configured database connections.
"""

from typing import Any, Sequence

import pandas as pd
from pydantic import BaseModel
from smolagents import Tool
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from ai_core.prompts import dedent_ws
from utils.sql_utils import check_dsn_update_driver


class TableConfig(BaseModel):
    """Configuration for a database table."""

    name: str
    description: str | None = None


class SQLTool(Tool):
    """A tool for executing SQL queries on databases.

    This tool provides SQL query execution capabilities for AI agents,
    connecting to databases using DSN configuration and providing detailed
    schema information for configured tables.
    """

    name: str
    description: str
    inputs = {"query": {"type": "string", "description": "SQL query to execute"}}
    output_type = "object"
    dsn: str
    tables: list[TableConfig]
    engine: Engine | None = None

    def __init__(self, name: str, dsn: str, description: str, tables: dict[str, Any] | list[str] | None = None) -> None:
        """Initialize SQL tool with database connection and table configuration.

        Args:
            dsn: Database connection string
            description: Description of the database purpose
            tables: Table configuration (dict with name/description or list of names)
            name: Tool name (defaults to 'SQL')
        """
        super().__init__()
        self.name = name
        self.dsn = dsn
        self.tables = self._parse_tables_config(tables)
        self._initialize_engine()
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
        base_desc = dedent_ws(f"""
            This tool executes SQL queries on a database. The ourput is a Pandas Dataframe. \n
            The database is described as:  {database_description}""")

        if not self.tables:
            return base_desc

        schema_info = self._get_detailed_schema_info()
        if schema_info:
            return f"{base_desc}\n\n{schema_info}"
        else:
            return base_desc

    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine using DSN."""
        try:
            dsn = check_dsn_update_driver(self.dsn, None)  # Async not supported
            self.engine = create_engine(dsn)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database using DSN '{self.dsn}': {e}") from e

    def _get_detailed_schema_info(self) -> str:
        """Get detailed schema information for configured tables.

        Returns:
            Formatted schema information suitable for LLM consumption
        """
        if not self.engine or not self.tables:
            return ""

        try:
            inspector = inspect(self.engine)
            available_tables = inspector.get_table_names()

            schema_parts = ["Available tables:"]

            for table_config in self.tables:
                table_name = table_config.name
                if table_name not in available_tables:
                    schema_parts.append(f"\n- {table_name}: Table not found in database")
                    continue

                # Get column information first to build description
                try:
                    columns = inspector.get_columns(table_name)
                    if not columns:
                        schema_parts.append(f"\n- {table_name}: No column information available")
                        continue

                    # Build comprehensive table description
                    table_desc = f"\n- Table '{table_name}':"

                    # Add user-provided description if available
                    if table_config.description:
                        table_desc += f"\n  Description: {table_config.description}"

                    # Generate helpful schema description for LLM
                    table_desc += self._generate_table_description(table_name, columns)

                    # Add detailed column information
                    table_desc += "\n  Schema:"
                    for col in columns:
                        col_type = str(col["type"]).replace("NULL", "VECTOR")  # Handle pgvector type
                        nullable = "nullable" if col["nullable"] else "not null"
                        table_desc += f"\n    - {col['name']}: {col_type} ({nullable})"

                    schema_parts.append(table_desc)

                except Exception as e:
                    schema_parts.append(f"\n- {table_name}: Could not retrieve schema information ({e})")

            return "\n".join(schema_parts)

        except Exception:
            return "Schema information not available"

    def _generate_table_description(self, table_name: str, columns: Sequence[Any]) -> str:
        """Generate helpful table description for LLM based on schema analysis.

        Args:
            table_name: Name of the table
            columns: Sequence of column information from SQLAlchemy inspector

        Returns:
            Generated description text
        """
        description_parts = []

        # Analyze columns to infer purpose
        column_names = [col["name"].lower() for col in columns]
        column_types = [str(col["type"]).upper() for col in columns]

        # Identify key columns and their purposes (avoid duplicates)
        key_columns = []
        processed_columns = set()

        # Process columns in order of specificity to avoid duplicates
        for col in columns:
            col_name = col["name"]
            col_name_lower = col_name.lower()

            if col_name in processed_columns:
                continue

            if "embedding" in col_name_lower:
                key_columns.append(f"'{col_name}' (vector embeddings)")
                processed_columns.add(col_name)
            elif "tsv" in col_name_lower:
                key_columns.append(f"'{col_name}' (full-text search)")
                processed_columns.add(col_name)
            elif "content" in col_name_lower:
                key_columns.append(f"'{col_name}' (text content)")
                processed_columns.add(col_name)
            elif "id" in col_name_lower:
                key_columns.append(f"'{col_name}' (identifier)")
                processed_columns.add(col_name)
            elif "metadata" in col_name_lower:
                key_columns.append(f"'{col_name}' (metadata)")
                processed_columns.add(col_name)

        # Build description
        description_parts.append(f"\n  Purpose: This table contains {len(columns)} columns")

        if key_columns:
            description_parts.append(f"\n  Key columns: {', '.join(key_columns)}")

        # Add query guidance
        guidance = []
        if any("embedding" in name for name in column_names):
            guidance.append("supports vector similarity searches")
        if any("tsv" in name for name in column_names):
            guidance.append("supports full-text search queries")
        if any("id" in name for name in column_names):
            guidance.append("use ID fields for filtering and joins")

        if guidance:
            description_parts.append(f"\n  Query guidance: This table {', '.join(guidance)}")

        return "".join(description_parts)

    def forward(self, query: str) -> pd.DataFrame:  # type: ignore
        """Execute SQL query and return results as DataFrame.

        Args:
            query: SQL query string to execute

        Returns:
            DataFrame containing query results
        """
        if not query or not query.strip():
            raise ValueError("SQL query cannot be empty")

        if not self.engine:
            self._initialize_engine()

        try:
            # Execute query and return as DataFrame
            with self.engine.connect() as conn:  # type: ignore
                result = pd.read_sql_query(text(query), conn)
                return result
        except Exception as e:
            raise RuntimeError(f"Failed to execute SQL query: {e}") from e

    def get_schema_info(self) -> str:
        """Get database schema information for context.

        Returns:
            String representation of configured tables and their schemas
        """
        return self._get_detailed_schema_info() or "No schema information available"
