"""
Module to improve interoperability between LangGraph CompiledGraph and Functional API,
and development of custom ReAct Agents.
Rationale is that LangChain recommend to use the prebuilt ReactAgent first, and then move to
a custom implementation.

The module provides tools for creating and managing LangGraph sessions with LLMs, including:
- LangGraphSession class for managing agent sessions
- Helper functions for streaming and printing responses
- Custom React agent creation

Example usage:
```python
# Create session with memory checkpointing
checkpointer = MemorySaver()
llm = get_llm()
session = LangGraphSession(checkpointer=checkpointer, thread_id="12345", llm=llm)

# Create prebuilt ReAct agent with tools
tools = [get_weather]
session.create_prebuilt_react_agent(tools)

# Run queries
result = await session.ainvoke("What's the weather in NYC?")
stream = session.stream("What's the weather in NYC?")

# Create custom React Agent based on Functional API
custom_agent = create_custom_react_agent(llm, tools, checkpointer)
session.set_graph(custom_agent)
...
#
```
"""

from typing import AsyncIterator, Iterator, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, ConfigDict


class LangGraphSession(BaseModel):
    """Manages a LangGraph session with checkpointing and tools.

    Handles creation and execution of LangGraph agents with support for:
    - Memory checkpointing
    - Streaming responses
    - Custom tools
    - Prebuilt and custom agent graphs

    Attributes:
        checkpointer: Checkpoint storage for session state
        thread_id: Unique identifier for the session
        llm: Language model to use
        extra_config: Additional configuration options
    """

    checkpointer: BaseCheckpointSaver
    thread_id: str
    llm: BaseChatModel
    extra_config: dict = {}
    system_prompt: str = ""
    prebuilt_react_agent: bool = False
    name: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _graph: CompiledGraph | None = None

    def config(self, call_conf: dict) -> RunnableConfig:
        config = {"configurable": {"thread_id": self.thread_id}} | self.extra_config | call_conf
        return cast(RunnableConfig, config)

    def create_prebuilt_react_agent(self, tools: list[BaseTool]) -> None:
        """Create a prebuilt ReAct agent with the given tools."""
        self._graph = create_react_agent(
            self.llm, tools, checkpointer=self.checkpointer, prompt=self.system_prompt or None
        )
        self.prebuilt_react_agent = True

    def set_graph(self, graph):
        """Set a custom graph for the session."""
        self._graph = graph

    async def ainvoke(self, query: str, call_config: dict = {}) -> AIMessage:
        """Execute a query asynchronously and return the final response.

        Example:
        ```python
        result = await session.ainvoke("What's the weather in NYC?")
        ```
        """
        assert self._graph

        # Not sure of these tests to make difference between
        if self.prebuilt_react_agent:
            inputs = {"messages": [("user", query)]}
            response = await self._graph.ainvoke(inputs, config=self.config(call_config))

        elif isinstance(self._graph, CompiledGraph):  # NOT TESTED
            inputs = {"messages": [("user", query)] + [("system", self.system_prompt)]}
            response = await self._graph.ainvoke(inputs, config=self.config(call_config))
        else:
            inputs = [{"role": "user", "content": query}, {"role": "system", "content": self.system_prompt}]
            response = await self._graph.ainvoke(inputs, config=self.config(call_config))  # type: ignore
        if isinstance(response, AIMessage):
            return response
        else:
            return response["messages"][-1]

    def stream(self, query: str, call_config: dict = {}) -> Iterator:
        """Stream the response to a query.
        Example:
        ```python
        stream = session.stream("What's the weather in NYC?")
        print_stream(stream)
        ```
        """
        assert self._graph
        if self.prebuilt_react_agent:
            inputs = {"messages": [("user", query)]}
            return self._graph.stream(inputs, config=self.config(call_config))
        elif isinstance(self._graph, CompiledGraph):
            inputs = {"messages": [("user", query)] + [("system", self.system_prompt)]}
            return self._graph.stream(inputs, config=self.config(call_config))
        else:
            inputs = [{"role": "user", "content": query}, {"role": "system", "content": self.system_prompt}]
            return self._graph.stream(inputs, config=self.config(call_config), stream_mode="values")

    async def astream(self, query: str, call_config: dict = {}) -> AsyncIterator:
        """Stream the response to a query.
        Example:
        ```python
        stream = session.stream("What's the weather in NYC?")
        print_stream(stream)
        ```
        """
        assert self._graph
        if self.prebuilt_react_agent:
            inputs = {"messages": [("user", query)]}
            return self._graph.astream(inputs, config=self.config(call_config))
        elif isinstance(self._graph, CompiledGraph):
            inputs = {"messages": [("user", query)] + [("system", self.system_prompt)]}
            return self._graph.astream(inputs, config=self.config(call_config))
        else:
            inputs = [{"role": "user", "content": query}, {"role": "system", "content": self.system_prompt}]
            return self._graph.astream(inputs, config=self.config(call_config), stream_mode="update, custom")
