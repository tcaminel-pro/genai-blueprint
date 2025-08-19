"""
Common code for Langgraph
"""

from typing import Any, AsyncIterator, Iterator

from langchain_core.messages import AIMessage

from src.utils.markdown import looks_like_markdown


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
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()

    if isinstance(step, AIMessage):
        content = step.content if details else "AI Message"
        console.print(
            Panel(
                content,
                title="[bold white on royal_blue1] AI Message [/bold white on royal_blue1]",
                border_style="royal_blue1",
                style="bold white on royal_blue1",
            )
        )

    elif isinstance(step, dict):
        for node, updates in step.items():
            if "messages" in updates:
                message_repr = updates["messages"][-1].pretty_repr()
                title_line = message_repr.split("\n")[0]
                body = "\n".join(message_repr.split("\n")[1:]) if "\n" in message_repr else ""
                # Check if this is an AI Message to apply special styling
                if "Ai Message" in title_line:
                    title = f"[white on color(17)] {title_line} [/white on color(17)]"
                    style = "white on color(17)"
                else:
                    title = f"[bold blue]{title_line}[/bold blue]"
                    style = ""

                if looks_like_markdown(body):
                    console.print(Panel(Markdown(body), title=title, border_style="bright_blue", style=style))
                else:
                    console.print(Panel(body, title=title, border_style="bright_blue", style=style))
            else:
                title = f"[bold blue]Update from: '{node}'[/bold blue]"
                content = updates if details else str(type(updates))
                if looks_like_markdown(str(content)):
                    console.print(Panel(Markdown(str(content)), title=title, border_style="blue"))
                else:
                    console.print(Panel(content, title=title, border_style="blue"))

    elif isinstance(step, tuple):
        step_type, content = step
        console.print(Panel(f"[bold magenta]Step Type: {step_type}[/bold magenta]", border_style="magenta"))

        for node, updates in content.items():
            if "messages" in updates:
                message_repr = updates["messages"][-1].pretty_repr()
                title_line = message_repr.split("\n")[0]
                title = f"[bold yellow]{title_line}[/bold yellow]"
                body = "\n".join(message_repr.split("\n")[1:]) if "\n" in message_repr else ""
                if _is_likely_markdown(body):
                    console.print(Panel(Markdown(body), title=title.upper(), border_style="blue"))
                else:
                    console.print(Panel(body, title=title.upper(), border_style="blue"))
            else:
                title = f"[bold yellow]Update from: {node}[/bold yellow]"
                detail_content = updates if details else str(type(updates).__name__)
                if _is_likely_markdown(str(detail_content)):
                    console.print(Panel(Markdown(str(detail_content)), title=title, border_style="yellow"))
                else:
                    console.print(Panel(detail_content, title=title, border_style="yellow"))

    else:
        console.print(Panel(str(step), title="[dim]Step[/dim]", border_style="white"))
    console.print()
