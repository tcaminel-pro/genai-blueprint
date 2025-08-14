"""
Common code for Langgraph
"""

from typing import Any, AsyncIterator, Iterator

from langchain_core.messages import AIMessage
from rich import print


def stream_node_response_content(stream: Iterator, node: str = "agent") -> Iterator:
    """Stream response content from a specific node in the graph.

    Args:
        stream: The output stream from LangGraph
        node: The node name to filter responses from

    Example : use with streamlit:
    ```python
        stream = lg_session.stream(query, {"callbacks": [st_callback]})
        text_response = st.chat_message("assistant").write_stream(stream_node_response_content(stream))
    """
    for chunk in stream:
        if isinstance(chunk, AIMessage):
            yield chunk.content
        else:
            for resp_node, updates in chunk.items():
                if resp_node == node:
                    if "messages" in updates:
                        yield (updates["messages"][-1].content)
                    else:
                        yield (str(updates))


async def astream_node_response_content(stream: AsyncIterator, node: str = "agent") -> AsyncIterator:
    """Stream response content from a specific node in the graph.

    Args:
        stream: The output stream from LangGraph
        node: The node name to filter responses from

    Example : use with streamlit:
    ```python
        stream = lg_session.stream(query, {"callbacks": [st_callback]})
        text_response = st.chat_message("assistant").write_stream(astream_node_response_content(stream))
    ```

    Note: This function is an AsyncGenerator compatible with Streamlit's write_stream method.
    """
    async for chunk in stream:
        if isinstance(chunk, AIMessage):
            yield chunk.content
        else:
            for resp_node, updates in chunk.items():
                if resp_node == node:
                    if "messages" in updates:
                        yield (updates["messages"][-1].content)
                    else:
                        yield (str(updates))


def print_stream(stream: Iterator, content: bool = True) -> None:
    """Print streamed responses in a readable format."""
    for step in stream:
        print_step(step, content)


async def print_astream(stream: AsyncIterator, content: bool = True) -> None:
    """Print async streamed responses in a readable format."""
    async for step in stream:
        print_step(step, content)


def print_step(step: Any, details: bool = True) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.json import JSON
    from rich.text import Text

    console = Console()

    if isinstance(step, AIMessage):
        content = step.content if details else "AI Message"
        console.print(Panel(content, title="[bold green]AIMessage[/bold green]", border_style="green"))

    elif isinstance(step, dict):
        for node, updates in step.items():
            title = f"[bold blue]Update from: '{node}'[/bold blue]"
            if "messages" in updates:
                console.print(Panel.fit(Text.from_markup(title)))
                console.print(updates["messages"][-1].pretty_repr())
            else:
                content = updates if details else str(type(updates))
                console.print(Panel(content, title=title, border_style="blue"))

    elif isinstance(step, tuple):
        step_type, content = step
        console.print(Panel(f"[bold magenta]Step Type: {step_type}[/bold magenta]", border_style="magenta"))

        for node, updates in content.items():
            title = f"[bold yellow]Update from: {node}[/bold yellow]"
            if "messages" in updates:
                console.print(Panel.fit(Text.from_markup(title)))
                console.print(updates["messages"][-1].pretty_repr())
            else:
                detail_content = updates if details else str(type(updates).__name__)
                console.print(Panel(detail_content, title=title, border_style="yellow"))

    else:
        console.print(Panel(str(step), title="[dim]Step[/dim]", border_style="white"))
    console.print()
