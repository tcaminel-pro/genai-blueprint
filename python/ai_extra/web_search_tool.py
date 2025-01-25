"""
Web search tools for AI applications.

This module provides tools for performing web searches, with support for multiple
search providers including Tavily and DuckDuckGo. The tools are designed to be
used in AI workflows where external information retrieval is needed.

Key Features:
- Automatic provider selection based on API key availability
- Configurable result limits
- Unified interface for different search backends
"""

import os

from langchain_core.tools import tool


@tool
def basic_web_search(query: str) -> str:
    """Run web search on the question."""
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
        except ImportError as ex:
            raise ImportError(
                "tavily-python package is required. Install with: poetry add tavily-python --group demos"
            ) from ex

        tavily_tool = TavilySearchResults(max_results=5)
        docs = tavily_tool.invoke({"query": query})
        web_results = "\n".join([d["content"] for d in docs])
    else:
        from langchain_community.tools import DuckDuckGoSearchRun

        duckduck_search_tool = DuckDuckGoSearchRun()
        web_results = duckduck_search_tool.invoke(query)

    return web_results
