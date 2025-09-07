"""Additional tools for SmolAgents integration.

This module provides custom tools and utilities for use with SmolAgents,
including stock data retrieval, DataFrame operations, and historical data access.
"""

from pathlib import Path

import pandas as pd
from smolagents import Tool

from src.utils.load_data import load_tabular_data_once


class DataFrameTool(Tool):
    """A tool for working with Pandas DataFrames from various data sources.

    This tool provides access to tabular data stored in files and allows the AI agent
    to perform data analysis operations on the loaded DataFrame.

    Attributes:
        name: Unique identifier for the tool
        description: Brief description of the tool's functionality
        source_path: Path to the data file containing the DataFrame
    """

    name: str
    description: str
    inputs = {}
    output_type = "object"
    source_path: Path

    def __init__(self, name: str, description: str, source_path: Path) -> None:
        super().__init__()
        self.name = name
        self.description = f"This tool returns a Pandas DataFrame with content described as: '{description}'"
        self.source_path = Path(source_path)
        try:
            import pandas as pd  # noqa: F401
        except ImportError as e:
            raise ImportError("You must install package `pandas` to run this tool`.") from e
        if not self.source_path.exists():
            raise ValueError(f"Incorrect source file: {self.source_path}")

    def forward(self) -> pd.DataFrame:  # type: ignore
        """Load and return a DataFrame from the configured source path."""
        df = load_tabular_data_once(self.source_path)
        return df
