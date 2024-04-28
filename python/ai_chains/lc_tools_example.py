"""

Adapted from https://blog.langchain.dev/tool-calling-with-langchain/  
"""

from typing import Any, Callable

from devtools import debug
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_tool_calling_agent,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool

from python.ai_core.chain_registry import (
    RunnableItem,
    register_runnable,
    to_key_param_callable,
)
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
    llm = LlmFactory(llm_id=config["llm"]).get()
    return llm.bind_tools(tools)  # type: ignore


register_runnable(
    RunnableItem(
        tag="Tool",
        name="Calculator tool",
        runnable=create_runnable,
        examples=["what's 5 raised to the 3"],
    )
)


def create_executor(config: dict) -> Runnable:
    debug(config)
    llm = LlmFactory(llm_id=config["llm"]).get()

    if False:
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, prompt=prompt)  # type: ignore
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "you're a helpful assistant"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # NOT IMPLEMENTED on most LLM yet
        agent = create_tool_calling_agent(llm, tools, prompt)  # type: ignore


register_runnable(
    RunnableItem(
        tag="Agent",
        name="Calculator agent",
        runnable=to_key_param_callable("input", create_executor),
        examples=["what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241"],
    )
)
