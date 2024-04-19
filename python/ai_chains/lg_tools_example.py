"""

Adapted from https://blog.langchain.dev/tool-calling-with-langchain/  
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor


from python.ai_core.chain_registry import register_runnable
from python.ai_core.llm import llm_getter


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


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_with_tools = llm.bind_tools(tools)


register_runnable(
    "Tool",
    "Calculator tool",
    llm_with_tools,
    examples=["what's 5 raised to the 2.743"],
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you're a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

register_runnable(
    "Agent",
    "Calculator agent",
    agent_executor,
    examples=["what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241"],
)


# agent_executor.invoke(
#     {
#         "input": "what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241",
#     }
# )
