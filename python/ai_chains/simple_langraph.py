import json

from langchain_core.messages import FunctionMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolInvocation


# Define the function that determines whether to continue or finish
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("agent", should_continue)
workflow.add_node("model_call", call_model)
workflow.add_node("tool_call", call_tool)

# Add conditional edge based on agent's decision
workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "model_call", "end": "END"}
)

# Set the entry point as the agent node
workflow.set_entry_point("agent")
