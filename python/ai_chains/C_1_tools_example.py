"""

Adapted from https://blog.langchain.dev/tool-calling-with-langchain/  
"""


from langchain import hub
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

from python.ai_core.chain_registry import (
    Example,
    RunnableItem,
    register_runnable,
)
from python.ai_core.llm import get_llm, get_llm_info


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

tavily_tool = TavilySearchResults(max_results=5)

tools = [tavily_tool]


# Whenever we invoke `llm_with_tool`, all three of these tool definitions
# are passed to the model.


def create_runnable(config: dict) -> Runnable:
    llm = get_llm(llm_id=config["llm"])
    return llm.bind_tools(tools)  # type: ignore


register_runnable(
    RunnableItem(
        tag="Tool",
        name="Calculator tool",
        runnable=create_runnable,
        examples=[Example(query=["what's 5 raised to the 3"])],
    )
)


def create_executor(config: dict) -> Runnable:
    llm_id = config["llm"]
    llm = get_llm(llm_id)
    info = get_llm_info(llm_id)

    prompt = hub.pull(info.agent_builder.hub_prompt)

    debug(prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful Search Assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent_creator = info.agent_builder.create_function
    debug(agent_creator)

    agent = agent_creator(llm, tools, prompt)  # type: ignore
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # type: ignore
    return agent_executor


register_runnable(
    RunnableItem(
        tag="Agent",
        name="Calculator agent",
        runnable=("input", create_executor),
        examples=[
            Example(
                query=[
                    "what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241"
                ]
            )
        ],
    )
)
