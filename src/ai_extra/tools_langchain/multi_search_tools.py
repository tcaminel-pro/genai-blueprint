"""
Search Tools for Deep Agents

This module provides real web search capabilities for deep agents,
supporting multiple search providers based on available API keys.
"""

import os

from langchain.tools import tool
from loguru import logger


def create_search_function(verbose: bool = False):
    """
    Create the best available search function based on available API keys.
    Returns a raw Python function that can be used directly with deep agents.

    Priority order:
    1. Tavily (if TAVILY_API_KEY is set)
    2. Serper (if SERPER_API_KEY is set)
    3. DuckDuckGo (no API key required)
    4. Mock search (fallback)

    Args:
        verbose: Whether to print which search provider is being used

    Returns:
        A raw search function that can be used with deep agents
    """

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    serper_api_key = os.environ.get("SERPER_API_KEY")

    # Try Tavily first (best option)
    if tavily_api_key:
        try:
            from tavily import TavilyClient

            tavily_client = TavilyClient(api_key=tavily_api_key)

            def internet_search(
                query: str,
                max_results: int = 5,
                topic: str = "general",
                include_raw_content: bool = False,
            ):
                """Run a web search using Tavily API"""
                try:
                    return tavily_client.search(
                        query,
                        max_results=max_results,
                        include_raw_content=include_raw_content,
                        topic=topic,
                    )
                except Exception as e:
                    logger.error(f"Tavily search error: {e}")
                    return {"error": str(e)}

            if verbose:
                logger.info("Using Tavily for web search")
            return internet_search

        except ImportError:
            logger.warning("Tavily client not installed. Install with: pip install tavily-python")

    # Try Serper as second option
    if serper_api_key:
        try:
            import requests

            def internet_search(
                query: str,
                max_results: int = 5,
                topic: str = "general",
                include_raw_content: bool = False,
            ):
                """Run a web search using Serper API"""
                try:
                    endpoint = "news" if topic == "news" else "search"
                    url = f"https://google.serper.dev/{endpoint}"
                    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

                    response = requests.post(url, headers=headers, json={"q": query, "num": max_results})

                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("organic", []) if endpoint == "search" else data.get("news", [])

                        # Format to match Tavily's response structure
                        formatted_results = {
                            "results": [
                                {
                                    "title": r.get("title", "N/A"),
                                    "url": r.get("link", "N/A"),
                                    "content": r.get("snippet", "N/A"),
                                }
                                for r in results[:max_results]
                            ]
                        }
                        return formatted_results
                    else:
                        return {"error": f"Search API error: {response.status_code}"}

                except Exception as e:
                    logger.error(f"Serper search error: {e}")
                    return {"error": str(e)}

            if verbose:
                logger.info("Using Serper for web search")
            return internet_search

        except ImportError:
            logger.warning("Requests library not available")

    # Try DuckDuckGo (no API key required)
    try:
        from duckduckgo_search import DDGS

        def internet_search(
            query: str,
            max_results: int = 5,
            topic: str = "general",
            include_raw_content: bool = False,
        ):
            """Run a web search using DuckDuckGo"""
            try:
                with DDGS() as ddgs:
                    if topic == "news":
                        results = list(ddgs.news(query, max_results=max_results))
                    else:
                        results = list(ddgs.text(query, max_results=max_results))

                    # Format to match Tavily's response structure
                    formatted_results = {
                        "results": [
                            {
                                "title": r.get("title", "N/A"),
                                "url": r.get("href", r.get("url", "N/A")),
                                "content": r.get("body", r.get("description", "N/A")),
                            }
                            for r in results
                        ]
                    }
                    return formatted_results

            except Exception as e:
                logger.error(f"DuckDuckGo search error: {e}")
                return {"error": str(e)}

        if verbose:
            logger.info("Using DuckDuckGo for web search (no API key)")
        return internet_search

    except ImportError:
        pass

    # Fallback to mock search
    def internet_search(
        query: str,
        max_results: int = 5,
        topic: str = "general",
        include_raw_content: bool = False,
    ):
        """Mock web search (no real search API available)"""
        return {
            "results": [
                {
                    "title": f"Mock result 1 about {query}",
                    "url": "https://example.com/1",
                    "content": f"This is a simulated search result about {query}. To enable real search, set TAVILY_API_KEY or install duckduckgo-search.",
                },
                {
                    "title": f"Mock result 2 about {query}",
                    "url": "https://example.com/2",
                    "content": f"Another simulated result related to {query}. Set environment variables for real results.",
                },
            ],
            "warning": "Using mock search. Set TAVILY_API_KEY or SERPER_API_KEY for real results, or install duckduckgo-search.",
        }

    if verbose:
        logger.warning("Using mock search. Set TAVILY_API_KEY or SERPER_API_KEY for real results")
    return internet_search


def create_search_tool(verbose: bool = False):
    """
    Create the best available search tool based on available API keys.
    Returns a LangChain tool for compatibility.

    Priority order:
    1. Tavily (if TAVILY_API_KEY is set)
    2. Serper (if SERPER_API_KEY is set)
    3. DuckDuckGo (no API key required)
    4. Mock search (fallback)

    Args:
        verbose: Whether to print which search provider is being used

    Returns:
        A LangChain tool function that can be used with LangChain agents
    """

    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    serper_api_key = os.environ.get("SERPER_API_KEY")

    # Try Tavily first (best option)
    if tavily_api_key:
        try:
            from tavily import TavilyClient

            tavily_client = TavilyClient(api_key=tavily_api_key)

            @tool
            def web_search(query: str, search_type: str = "general") -> str:
                """
                Search the web using Tavily API for real-time, high-quality results.

                Args:
                    query: Search query
                    search_type: Type of search - 'general' or 'news'

                Returns:
                    Formatted search results
                """
                try:
                    # Determine topic based on search type
                    topic = "news" if search_type == "news" else "general"

                    # Perform search
                    results = tavily_client.search(query, max_results=5, topic=topic, include_raw_content=False)

                    # Format results
                    formatted_results = []
                    for idx, r in enumerate(results.get("results", []), 1):
                        formatted_results.append(
                            f"{idx}. {r.get('title', 'N/A')}\n"
                            f"   URL: {r.get('url', 'N/A')}\n"
                            f"   {r.get('content', 'N/A')[:300]}...\n"
                        )

                    if formatted_results:
                        return f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
                    else:
                        return f"No results found for '{query}'"

                except Exception as e:
                    logger.error(f"Tavily search error: {e}")
                    return f"Search error: {str(e)}"

            if verbose:
                logger.info("Using Tavily for web search")
            return web_search

        except ImportError:
            logger.warning("Tavily client not installed. Install with: pip install tavily-python")

    # Try Serper as second option
    if serper_api_key:
        try:
            import requests

            @tool
            def web_search(query: str, search_type: str = "general") -> str:
                """
                Search the web using Serper API for Google search results.

                Args:
                    query: Search query
                    search_type: Type of search - 'general' or 'news'

                Returns:
                    Formatted search results
                """
                try:
                    endpoint = "news" if search_type == "news" else "search"
                    url = f"https://google.serper.dev/{endpoint}"
                    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

                    response = requests.post(url, headers=headers, json={"q": query, "num": 5})

                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("organic", []) if endpoint == "search" else data.get("news", [])

                        formatted_results = []
                        for idx, r in enumerate(results[:5], 1):
                            formatted_results.append(
                                f"{idx}. {r.get('title', 'N/A')}\n"
                                f"   URL: {r.get('link', 'N/A')}\n"
                                f"   {r.get('snippet', 'N/A')}\n"
                            )

                        if formatted_results:
                            return f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
                        else:
                            return f"No results found for '{query}'"
                    else:
                        return f"Search API error: {response.status_code}"

                except Exception as e:
                    logger.error(f"Serper search error: {e}")
                    return f"Search error: {str(e)}"

            if verbose:
                logger.info("Using Serper for web search")
            return web_search

        except ImportError:
            logger.warning("Requests library not available")

    # Try DuckDuckGo (no API key required)
    try:
        from duckduckgo_search import DDGS

        @tool
        def web_search(query: str, search_type: str = "general") -> str:
            """
            Search the web using DuckDuckGo (no API key required).

            Args:
                query: Search query
                search_type: Type of search - 'general' or 'news'

            Returns:
                Formatted search results
            """
            try:
                with DDGS() as ddgs:
                    if search_type == "news":
                        results = list(ddgs.news(query, max_results=5))
                    else:
                        results = list(ddgs.text(query, max_results=5))

                    formatted_results = []
                    for idx, r in enumerate(results, 1):
                        title = r.get("title", "N/A")
                        url = r.get("href", r.get("url", "N/A"))
                        content = r.get("body", r.get("description", "N/A"))

                        formatted_results.append(f"{idx}. {title}\n   URL: {url}\n   {content[:300]}...\n")

                    if formatted_results:
                        return f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
                    else:
                        return f"No results found for '{query}'"

            except Exception as e:
                logger.error(f"DuckDuckGo search error: {e}")
                return f"Search error: {str(e)}"

        if verbose:
            logger.info("Using DuckDuckGo for web search (no API key)")
        return web_search

    except ImportError:
        pass

    # Fallback to mock search
    @tool
    def web_search(query: str, search_type: str = "general") -> str:
        """
        Mock web search (no real search API available).

        To enable real search, set one of these environment variables:
        - TAVILY_API_KEY (recommended)
        - SERPER_API_KEY
        Or install duckduckgo-search for free search: pip install duckduckgo-search

        Args:
            query: Search query
            search_type: Type of search - 'general' or 'news'

        Returns:
            Mock search results with instructions
        """
        return (
            f"⚠️ Mock Search Results for: '{query}'\n\n"
            "This is a simulated search. To enable real web search:\n\n"
            "Option 1: Set TAVILY_API_KEY environment variable\n"
            "   - Get free API key at https://tavily.com\n"
            "   - Add to .env: TAVILY_API_KEY=your-key\n\n"
            "Option 2: Set SERPER_API_KEY environment variable\n"
            "   - Get API key at https://serper.dev\n"
            "   - Add to .env: SERPER_API_KEY=your-key\n\n"
            "Option 3: Use DuckDuckGo (free, no API key)\n"
            "   - Install: pip install duckduckgo-search\n\n"
            f"Mock results:\n"
            f"1. Sample result about {query}\n"
            f"2. Another result related to {query}\n"
            f"3. More information on {query}"
        )

    if verbose:
        logger.warning("Using mock search. Set TAVILY_API_KEY or SERPER_API_KEY for real results")
    return web_search


def test_search_tool():
    """Test the search tool to see which provider is being used."""
    search_tool = create_search_tool(verbose=True)

    # Test search
    result = search_tool.invoke({"query": "Atos news September 2025", "search_type": "news"})
    print(result)

    return search_tool


if __name__ == "__main__":
    test_search_tool()
