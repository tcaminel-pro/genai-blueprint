"""
Custom ReAct Agent based on Functional API

taken from https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch-functional/

"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages


def create_custom_react_agent(model: BaseChatModel, tools: list[BaseTool], checkpointer: BaseCheckpointSaver):
    """Create a custom ReAct agent from scratch using Functional API.

    Args:
        model: Language model to use
        tools: List of tools the agent can use
        checkpointer: Checkpoint storage for agent state
    Example:
    ```python
    tools = [get_weather]
    checkpointer = MemorySaver()
    llm = get_llm()
    custom_agent = create_custom_react_agent(llm, tools, checkpointer)
    ```
    """
    tools_by_name = {tool.name: tool for tool in tools}

    @task
    def call_model(messages):
        """Call model with a sequence of messages."""
        response = model.bind_tools(tools).invoke(messages)
        return response

    @task
    def call_tool(tool_call):
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        return ToolMessage(content=observation, tool_call_id=tool_call["id"])

    @entrypoint(checkpointer=checkpointer)
    def agent(messages, previous):
        if isinstance(messages, str):
            messages = [messages]
        if previous is not None:
            messages = add_messages(previous, messages)

        llm_response = call_model(messages).result()
        while True:
            if not llm_response.tool_calls:  # type: ignore
                break

            # Execute tools
            tool_result_futures = [call_tool(tool_call) for tool_call in llm_response.tool_calls]  # type: ignore
            tool_results = [fut.result() for fut in tool_result_futures]

            # Append to message list
            messages = add_messages(messages, [llm_response, *tool_results])

            # Call model again
            llm_response = call_model(messages).result()

        # Generate final response
        messages = add_messages(messages, llm_response)
        return entrypoint.final(value=llm_response, save=messages)

    return agent
