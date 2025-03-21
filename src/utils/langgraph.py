"""
Common code for Langgraph
"""

from typing import Any, AsyncIterator, Iterator

from devtools import pprint
from langchain_core.messages import AIMessage


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


async def astream_node_response_content(stream: AsyncIterator, node: str = "agent"):
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


def print_stream(stream: Iterator, content: bool = True):
    """Print streamed responses in a readable format."""
    for step in stream:
        print_step(step, content)


async def print_astream(stream: AsyncIterator, content: bool = True):
    """Print async streamed responses in a readable format."""
    async for step in stream:
        print_step(step, content)


def print_step(step: Any, details: bool = True) -> None:
    if isinstance(step, AIMessage):
        if details:
            print(step.content)
        else:
            print("AI Message")
    elif isinstance(step, dict):
        for node, updates in step.items():
            print(f"Update from: '{node}'")
            if "messages" in updates:
                updates["messages"][-1].pretty_print()
            else:
                if details:
                    pprint(updates)
                else:
                    print(type(updates))

    elif isinstance(step, tuple):
        #        debug(step)
        step_type, content = step
        print(f"step type: {step_type}")
        for node, updates in content.items():
            print(f"Update from: {node}")
            if "messages" in updates:
                updates["messages"][-1].pretty_print()
            else:
                if details:
                    pprint(updates)
                    # print(content)
                else:
                    print(type(updates).__name__)
    else:
        print(str(step))
    print("\n")
