"""

Adapted from https://blog.langchain.dev/tool-calling-with-langchain/  
"""

from typing import Any, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import Runnable, RunnableLambda
from python.ai_core.chain_registry import (
    RunnableItem,
    register_runnable,
    to_key_param_callable,
)

from devtools import debug

from python.ai_core.llm import LlmFactory  # ignore


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y


tools = [multiply, exponentiate, add]


# Whenever we invoke `llm_with_tool`, all three of these tool definitions
# are passed to the model.


def create_runnable(config: dict) -> Runnable:
    llm = LlmFactory()._get()
    return llm.bind_tools(tools)  # type: ignore


register_runnable(
    RunnableItem(
        tag="Tool",
        name="Calculator tool",
        runnable=create_runnable,
        examples=["what's 5 raised to the 3"],
    )
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you're a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def create_executor(conf: dict) -> Runnable:
    debug(conf)
    model = conf["llm"]
    llm = llm_factory(model)
    agent = create_tool_calling_agent(llm, tools, prompt)  # type: ignore
    return AgentExecutor(agent=agent, tools=tools)  # type: ignore


register_runnable(
    RunnableItem(
        tag="Agent",
        name="Calculator agent",
        runnable=to_key_param_callable("input", create_executor),
        examples=["what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241"],
    )
)
