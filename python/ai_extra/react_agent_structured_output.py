"""

This module provides an implementation of a ReAct agent that can generate structured output.
    The agent is built using LangChain components and is designed to be flexible and reusable for various applications.
"""

from typing import Literal, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def create_react_structured_output_graph(
    llm: BaseChatModel, tools: list[BaseTool], out_model_class: Type[T]
) -> CompiledGraph:
    """
    Creates a compiled LangGraph graph for a ReAct agent that can generate structured output.

    This function sets up a state graph with an agent node, a tool node, and a respond node.
    The agent node calls the model, the tool node processes the tools, and the respond node
    constructs the final response from the tool calls.

    Strongly inspired from : https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/#define-graph

    Args:
        model (BaseChatModel): The language model to use for the agent.
        tools (list[BaseTool]): A list of tools the agent can use.
        out_model_class (Type[T]): Pydantic class that will be used to generate the final response.

    Returns:
        CompiledGraph: A compiled graph representing the ReAct agent workflow.

    Example :
    .. code-block:: python
        class WeatherResponse(BaseModel):
            temperature: float = Field(description="The temperature in fahrenheit")
            ...
        @tool
        def get_weather(city: Literal["nyc", "sf"]):
            "Use this to get weather information."
            ...
        llm = get_llm(...)
        graph = create_react_structured_output_graph(llm, [get_weather], WeatherResponse)
        answer = graph.invoke(input={"messages": [("human", "what's the weather in New York?")]})["final_response"]

    """

    class AgentState(MessagesState):
        # Inherit 'messages' key from MessagesState, which is a list of chat messages
        final_response: out_model_class

    tools_and_result = tools + [out_model_class]
    llm_with_response_tool = llm.bind_tools(tools_and_result, tool_choice="any")  # Force the model to use tools

    # Define the function that calls the model
    def call_model(state: AgentState):
        response = llm_with_response_tool.invoke(state["messages"])
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the function that responds to the user
    def respond(state: AgentState):
        # Construct the final answer from the arguments of the last tool call
        response = out_model_class(**state["messages"][-1].tool_calls[0]["args"])  # type: ignore
        return {"final_response": response}

    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState) -> Literal["respond", "continue"]:
        messages = state["messages"]
        last_message = messages[-1]
        # If there is only one tool call and it is the response tool call we respond to the user
        if len(last_message.tool_calls) == 1 and last_message.tool_calls[0]["name"] == out_model_class.__name__:  # type: ignore
            return "respond"
        # Otherwise we will use the tool node again  (TBC)
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("respond", respond)
    workflow.add_node("tools", ToolNode(tools_and_result))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "respond": "respond",
        },
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("respond", END)
    graph = workflow.compile()
    return graph
