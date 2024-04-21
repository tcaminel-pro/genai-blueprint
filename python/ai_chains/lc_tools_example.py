"""

Adapted from https://blog.langchain.dev/tool-calling-with-langchain/  
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import Runnable

from python.ai_core.chain_registry import RunnableItem, register_runnable
from python.ai_core.llm import llm_factory


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


def create_runnable(info: dict) -> Runnable:
    llm = llm_factory(info["llm"])
    return llm.bind_tools(tools)


register_runnable(
    RunnableItem(
        tag="Tool",
        name="Calculator tool",
        runnable=create_runnable,
        examples=["what's 5 raised to the 2.743"],
    )
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you're a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def create_executor(info: dict) -> Runnable:
    model = info["llm"]
    llm = llm_factory(model)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


register_runnable(
    RunnableItem(
        tag="Agent",
        name="Calculator agent",
        runnable=create_executor,
        examples=["what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241"],
        key="input",
    )
)
