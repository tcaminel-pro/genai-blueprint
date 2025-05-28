"""Streamlit page for ReAct Agent demo.

Provides an interactive interface to run ReAct agents with different configurations.
Supports custom tools, MCP servers integration, and demo presets.

"""

import asyncio
import uuid
from json import tool
from typing import List, Tuple, cast

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import tracing_v2_enabled
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.ai_core.llm import get_llm
from src.ai_core.mcp_client import get_mcp_servers_dict
from src.ai_core.prompts import dedent_ws, dict_input_message
from src.utils.config_mngr import global_config
from src.utils.streamlit.thread_issue_fix import get_streamlit_cb
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.streamlit_chat import StreamlitStatusCallbackHandler, display_messages

load_dotenv()

llm_config_widget(st.sidebar, False)


@tool
def my_custom_weather(location: str) -> str:
    """Return an approximate weather for given location.

    Args:
        location: City name to get weather for

    Returns:
        Weather description string
    """

    if location == "Toulouse":
        return "Il faut beau"
    else:
        return "I don't know"


st.title("ReAct Agent")

# Default system prompt
SYSTEM_PROMPT = dedent_ws(
    """
    Your are a helpful assistant. Use provided tools to answer questions. \n
    - If the user asks for a list of something and that the tool returns a list, print it as Markdown table. 
"""
)


# Define demo class
class ReactDemo(BaseModel):
    """Configuration for a ReAct agent demo preset.

    Attributes:
        name: Unique demo name
        tools: List of tool names to include
        mcp_servers: List of MCP server names to use
        examples: Example queries for the demo
        system_prompt: Custom system prompt for the agent
    """

    name: str
    tools: list[str] = []
    mcp_servers: list[str] = []
    examples: list[str] = []
    system_prompt: str = SYSTEM_PROMPT
    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_demos_from_config() -> List[ReactDemo]:
    """Load demo configurations from global config.

    Returns:
        List of ReactDemo instances loaded from config

    Raises:
        Exception: If config loading fails
    """
    try:
        demos_config = global_config().get_list("react_agent_demos")
        result = []
        # Create Demo objects from the configuration
        for demo_config in demos_config:
            name = demo_config.get("name", "")
            examples = demo_config.get("examples", [])
            mcp_servers = demo_config.get("mcp_servers", [])
            tools = demo_config.get("tools", [])
            system_prompt = demo_config.get("system_prompt", SYSTEM_PROMPT)

            demo = ReactDemo(
                name=name,
                tools=tools,
                mcp_servers=mcp_servers,
                examples=examples,
                system_prompt=system_prompt,
            )
            result.append(demo)
        return result
    except Exception as e:
        logger.exception(f"Error loading demos from config: {e}")
        return []


SAMPLES_DEMOS = load_demos_from_config()

debug(type(my_custom_weather))
local_tools = []
for demo in SAMPLES_DEMOS:
    for tool_name in demo.tools:
        if tool_name in globals() and callable(globals()[tool_name]):
            tool_func = globals()[tool_name]
            if isinstance(tool_func, BaseTool):
                local_tools.append(tool_func)


def clear_display() -> None:
    """Reset the chat display and tools state."""
    if "messages" in st.session_state:
        st.session_state.messages = []
    if "tools" in st.session_state:
        del st.session_state.tools


c01, c02 = st.columns([6, 4], border=False, gap="medium", vertical_alignment="top")
with c01.container(border=True):
    selected_pill = st.pills(
        "🎬 **Demos:**",
        options=[demo.name for demo in SAMPLES_DEMOS],
        default=SAMPLES_DEMOS[0].name,
        on_change=clear_display,
    )

# Get selected demo
demo = next((d for d in SAMPLES_DEMOS if d.name == selected_pill), None)
if demo is None:
    st.stop()

# Display demo information
col_display_left, col_display_right = st.columns([6, 3], vertical_alignment="bottom")
with col_display_right:
    if tools_list := ", ".join(f"'{t}'" for t in demo.tools):
        st.markdown(f"**Tools**: *{tools_list}*")
    if mcp_list := ", ".join(f"'{mcp}'" for mcp in demo.mcp_servers):
        st.markdown(f"**MCP**: *{mcp_list}*")

with col_display_left:
    sample_search = col_display_left.selectbox(
        label="Sample",
        placeholder="Select an example (optional)",
        options=demo.examples,
        index=None,
        label_visibility="collapsed",
    )

# Display tools if available
with st.expander("Available Tools", expanded=False):
    if tools := st.session_state.get("tools"):
        tools = cast(list[BaseTool], tools)
        d = {"Name": [t.name for t in tools], "Description": [t.description for t in tools]}
        st.dataframe(d)


@st.cache_resource()
def get_agent_config() -> tuple[RunnableConfig, BaseCheckpointSaver]:
    """Create and cache agent configuration.

    Returns:
        Tuple of (RunnableConfig, checkpoint saver) with unique thread ID
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    checkpointer = MemorySaver()
    return cast(RunnableConfig, config), checkpointer


async def main() -> None:
    """Main async function to run the ReAct agent demo.

    Handles:
    - UI setup and demo selection
    - Tool initialization
    - Agent execution
    - Streaming output display
    """
    display_messages(st)
    config, checkpointer = get_agent_config()
    llm = get_llm()

    # Get MCP servers from selected demo
    mcp_servers_params = get_mcp_servers_dict(demo.mcp_servers) if demo.mcp_servers else {}
    client = None
    try:
        client = MultiServerMCPClient(mcp_servers_params)
        rcp_tools = await client.get_tools()
        all_tools = local_tools + rcp_tools
        if "tools" not in st.session_state:
            st.session_state["tools"] = all_tools
            st.rerun()

        # Create agent with demo's system prompt
        agent = create_react_agent(model=llm, tools=all_tools, prompt=demo.system_prompt, checkpointer=checkpointer)

        # Use sample as default query if selected
        query = st.chat_input(placeholder="Enter your question...") or sample_search

        if query:
            st.session_state.messages.append(HumanMessage(content=query))
            st_callback = get_streamlit_cb(st.container())
            st.chat_message("human").write(query)

            # Create status to show agent progress
            status = st.status("Agent starting...", expanded=True)
            status_callback = StreamlitStatusCallbackHandler(status)

            inputs = dict_input_message(user=query)

            config["configurable"].update({"st_container": status})
            response = "A problem occurred"
            with tracing_v2_enabled() as cb:
                astream = agent.astream(
                    inputs,
                    config | {"callbacks": [st_callback, status_callback]},  # , stream_mode=["values", "custom"]
                )
                async for step in astream:
                    if isinstance(step, Tuple):  # we are likely with stream_mode = "updates", that generate tuple
                        step = step[1]
                    for node, update in step.items():
                        if node == "agent":
                            response = update["messages"][-1]
                            assert isinstance(response, AIMessage)
                            st.chat_message("ai").write(response.content)
                url = cb.get_run_url()
            status.update(label="Done", state="complete", expanded=False)
            st.session_state.messages.append(response)
            st.link_button("Trace", url)
    finally:
        if client:
            pass  # Add cleanup if needed


# Run the async main function
asyncio.run(main())
