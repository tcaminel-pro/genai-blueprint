import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# initialize server
mcp = FastMCP("tech_news")

USER_AGENT = "news-app/1.0"

NEWS_SITES = {"arstechnica": "https://arstechnica.com"}


async def fetch_news(url: str):
    """It pulls and summarizes the latest news from the specified news site."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs[:5]])
            return text
        except httpx.TimeoutException:
            return "Timeout error"


@mcp.tool()
async def get_tech_news(source: str):
    """
    Fetches the latest news from a specific tech news source.

    Args:
    source: Name of the news source (for example, "arstechnica" or "techcrunch").

    Returns:
    A brief summary of the latest news.
    """
    if source not in NEWS_SITES:
        raise ValueError(f"Source {source} is not supported.")

    news_text = await fetch_news(NEWS_SITES[source])
    return news_text


if __name__ == "__main__":
    mcp.run(transport="stdio")
