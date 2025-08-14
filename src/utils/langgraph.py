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
    from rich.pretty import Pretty
    
    console = Console()
    
    if isinstance(step, AIMessage):
        if details:
            console.print(Pretty(step.content))
        else:
            console.print("[bold]AI Message[/bold]")
    elif isinstance(step, dict):
        for node, updates in step.items():
            console.print(f"[bold cyan]Update from: '{node}'[/bold cyan]")
            if "messages" in updates:
                console.print(Pretty(updates["messages"][-1]))
            else:
                if details:
                    console.print(Pretty(updates))
                else:
                    console.print(f"[dim]{type(updates)}[/dim]")

    elif isinstance(step, tuple):
        step_type, content = step
        console.print(f"[bold magenta]step type: {step_type}[/bold magenta]")
        for node, updates in content.items():
            console.print(f"[bold cyan]Update from: {node}[/bold cyan]")
            if "messages" in updates:
                console.print(Pretty(updates["messages"][-1]))
            else:
                if details:
                    console.print(Pretty(updates))
                else:
                    console.print(f"[dim]{type(updates).__dim]")
    else:
        console.print(Pretty(step))
    console.print()
