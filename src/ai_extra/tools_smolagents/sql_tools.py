"""SQL tools for SmolAgents integration.

This module provides SQL database tools for use with SmolAgents,
allowing agents to execute SQL queries on configured database connections.
"""

import pandas as pd
from smolagents import Tool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from utils.sql_utils import check_dsn_update_driver


class SQLTool(Tool):
    """A tool for executing SQL queries on databases.

    This tool provides SQL query execution capabilities for AI agents,
    connecting to databases using DSN configuration from global config.

    Attributes:
        name: Unique identifier for the tool
        description: Brief description of the tool's functionality
        dsn_key: Configuration key for database DSN
        engine: SQLAlchemy engine for database connections
    """

    name: str
    description: str
    inputs = {"query": {"type": "string", "description": "SQL query to execute"}}
    output_type = "object"
    dsn: str
    engine: Engine | None = None

    def __init__(self, name: str, description: str, dsn: str) -> None:
        """Initialize SQL tool with database connection.

        Args:
            name: Unique name for this tool instance
            description: Description of the tool's purpose
            dsn: Configuration of database DSN
        """
        super().__init__()
        self.name = name
        self.description = f"This tool executes SQL queries on the database: '{description}'"
        self.dsn = dsn
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine using DSN from global config."""
        try:
            dsn = check_dsn_update_driver(self.dsn, None)  # Async not supported
            self.engine = create_engine(dsn)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database using DSN'{self.dsn}': {e}") from e

    def forward(self, query: str) -> pd.DataFrame:  # type: ignore
        """Execute SQL query and return results as DataFrame.

        Args:
            query: SQL query string to execute

        Returns:
            DataFrame containing query results

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If query execution fails
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
            String representation of available tables and columns
        """
        if not self.engine:
            self._initialize_engine()

        try:
            # Get table names
            with self.engine.connect() as conn:  # type: ignore
                tables_df = pd.read_sql_query(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"), conn
                )
                return f"Available tables: {', '.join(tables_df['table_name'].tolist())}"
        except Exception:
            # Fallback for databases that don't support information_schema
            return "Schema information not available for this database type"
