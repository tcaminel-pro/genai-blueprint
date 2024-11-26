"""
Commonly used tools 
"""

import os

from langchain_core.tools import tool

# TODO : Complete and improve !
# ex: Exe.ai API, Serper, ...


@tool
def basic_web_search(query: str) -> str:
    """Run web search on the question.  Call Tivaly if we have a key, DuckDucGo otherwise"""

    # TO BE COMPLETED to have similar behavior ...

    if os.environ.get("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults

        tavily_tool = TavilySearchResults(max_results=5)
        docs = tavily_tool.invoke({"query": query})
        web_results = "\n".join([d["content"] for d in docs])
    else:
        from langchain_community.tools import DuckDuckGoSearchRun

        duckduck_search_tool = DuckDuckGoSearchRun()
        web_results = duckduck_search_tool.invoke(query)

    return web_results
