# Taken from https://github.com/langchain-ai/langchain-mcp-adapters

# from loguru import logger
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@mcp.tool()
def exp(a: int, b: int) -> int:
    """Calculate a exponent b"""
    return a ^ b


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


def main() -> None:
    print("Start 'Math MCP Server")
    mcp.run(transport="stdio")

    # mcp.run_stdio_async


if __name__ == "__main__":
    main()
