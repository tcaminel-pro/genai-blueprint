"""Streamlit page for ReAct Agent demo.

Provides an interactive interface to run ReAct agents with different configurations.
Supports custom tools, MCP servers integration, and demo presets.

"""

import asyncio
import uuid
from pathlib import Path
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
from streamlit import session_state as sss

from src.ai_core.llm_factory import get_llm
from src.ai_core.mcp_client import get_mcp_servers_dict
from src.ai_core.prompts import dedent_ws, dict_input_message
from src.ai_extra.tools_langchain.shared_config_loader import LangChainAgentConfig, load_all_langchain_agent_configs
from src.utils.config_mngr import global_config
from src.utils.streamlit.thread_issue_fix import get_streamlit_cb
from src.webapp.ui_components.config_editor import edit_config_dialog
from src.webapp.ui_components.llm_selector import llm_selector_widget
from src.webapp.ui_components.streamlit_chat import StreamlitStatusCallbackHandler, display_messages

load_dotenv()

from langchain_community.tools import DuckDuckGoSearchRun  # noqa: E402

duckduck_search_tool = DuckDuckGoSearchRun()

CONFIG_FILE = "config/demos/react_agent.yaml"
assert Path(CONFIG_FILE).exists, f"Cannot load {CONFIG_FILE}"


@tool
def my_custom_weather(location: str) -> str:
    """Return an approximate weather in Toulouse

    Args:
        location: City name to get weather for

    Returns:
        Weather description string
    """

    if location == "Toulouse":
        return "Il faut beau"
    else:
        return "I don't know"


# Only run UI code when this script is executed in Streamlit context (not during import)
try:
    # This will only work when running in a Streamlit context
    _ = st.session_state  # This will raise an exception if not in Streamlit context
    
    st.title("ReAct Agent")
    
    # Sidebar: LLM selector and config editor (aligned with CodeAct Agent page)
    llm_selector_widget(st.sidebar)
    if st.sidebar.button(":material/edit: Edit Config", help="Edit anonymization configuration"):
        edit_config_dialog(CONFIG_FILE)
except (AttributeError, RuntimeError, Exception):
    # We're being imported, not running in Streamlit - skip UI code
    pass

# Default system prompt
SYSTEM_PROMPT = dedent_ws(
    """
    Your are a helpful assistant. Use provided tools to answer questions. \n
    - If the user asks for a list of something and that the tool returns a list, print it as Markdown table. 
"""
)


# Use shared config loader for consistent tool processing
# Only load demos when actually running the UI, not during import
SAMPLES_DEMOS = None


def clear_display() -> None:
    """Reset the chat display and tools state."""
    if "messages" in sss:
        sss.messages = []
    if "tools" in sss:
        del sss.tools


def display_header_and_demo_selector(sample_demos: list[LangChainAgentConfig]) -> str | None:
    """Displays the header and demo selector, returning the selected pill."""
    c01, c02 = st.columns([6, 4], border=False, gap="medium", vertical_alignment="top")
    c02.title(" ReAct Agent :material/smart_toy:")
    selected_pill = None

    if not sample_demos:
        with c01.container(border=True):
            st.warning("No demo configurations found. Please check your config file.")
        return None

    with c01.container(border=True):
        selector_col, edit_col = st.columns([8, 1], vertical_alignment="bottom")
        with selector_col:
            selected_pill = st.pills(
                ":material/open_with: **Demos:**",
                options=[demo.name for demo in sample_demos],
                default=sample_demos[0].name,
                on_change=clear_display,
            )
    return selected_pill


def display_demo_info_and_sample_selector(demo: LangChainAgentConfig, select_block):
    """Display demo information and sample selector."""
    col_display_left, col_display_right = select_block.columns([6, 3], vertical_alignment="bottom")
    with col_display_right:
        if demo.tools and (tools_list := ", ".join(f"'{t.name}'" for t in demo.tools)):
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
    return sample_search


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


def display_input_form(select_block, sample_search: str | None) -> tuple[str, bool]:
    """Displays the input form and returns user input."""
    with select_block.form("react_form", border=False):
        cf1, cf2 = st.columns([15, 1], vertical_alignment="bottom")
        prompt = cf1.text_area(
            "Your question",
            height=68,
            placeholder="Enter your question here...",
            value=sample_search or "",
            label_visibility="collapsed",
        )
        submitted = cf2.form_submit_button(label="", icon=":material/send:")
    return prompt, submitted


async def handle_agent_execution(placeholder, demo: LangChainAgentConfig, query: str) -> None:
    """Handle the agent execution with proper UI layout."""
    HEIGHT = 800
    exec_block = placeholder.container()
    col_display_left, col_display_right = exec_block.columns(2)
    chat_container = col_display_left.container(height=HEIGHT)
    result_display = col_display_right.container(height=HEIGHT)

    config, checkpointer = get_agent_config()
    llm = get_llm()

    # Get MCP servers from selected demo
    mcp_servers_params = get_mcp_servers_dict(demo.mcp_servers) if demo.mcp_servers else {}
    client = None
    try:
        client = MultiServerMCPClient(mcp_servers_params)
        rcp_tools = await client.get_tools()
        all_tools = demo.tools + rcp_tools
        if "tools" not in sss:
            sss["tools"] = all_tools

        # Create agent with demo's system prompt or default
        prompt = demo.system_prompt or SYSTEM_PROMPT
        agent = create_react_agent(model=llm, tools=all_tools, prompt=prompt, checkpointer=checkpointer)

        with chat_container:
            display_messages(st)
            st.chat_message("human").write(query)

            # Create status to show agent progress
            status = st.status("Agent starting...", expanded=True)
            status_callback = StreamlitStatusCallbackHandler(status)
            st_callback = get_streamlit_cb(st.container())

            inputs = dict_input_message(user=query)
            config["configurable"].update({"st_container": status})
            response = "A problem occurred"

            with tracing_v2_enabled() as cb:
                astream = agent.astream(
                    inputs,
                    config | {"callbacks": [st_callback, status_callback]},
                )
                async for step in astream:
                    if isinstance(step, Tuple):
                        step = step[1]
                    for node, update in step.items():
                        if node == "agent":
                            response = update["messages"][-1]
                            assert isinstance(response, AIMessage)
                            st.chat_message("ai").write(response.content)
                url = cb.get_run_url()

            status.update(label="Done", state="complete", expanded=False)
            if "messages" not in sss:
                sss.messages = []
            sss.messages.append(HumanMessage(content=query))
            sss.messages.append(response)
            st.link_button("Trace", url)

    finally:
        if client:
            pass  # Add cleanup if needed


async def main() -> None:
    """Main async function to run the ReAct agent demo."""
    # Load demos when actually running the UI
    global SAMPLES_DEMOS
    if SAMPLES_DEMOS is None:
        SAMPLES_DEMOS = load_all_langchain_agent_configs(CONFIG_FILE, "react_agent_demos")
    
    # Main UI setup
    selected_pill = display_header_and_demo_selector(SAMPLES_DEMOS)

    # Get selected demo
    demo = next((d for d in SAMPLES_DEMOS if d.name == selected_pill), None)
    if demo is None:
        st.stop()

    # Create main placeholder and select block container
    placeholder = st.empty()
    select_block = placeholder.container()

    # Display demo information and sample selector
    sample_search = display_demo_info_and_sample_selector(demo, select_block)

    # Display tools if available
    with select_block.expander("Available Tools", expanded=False):
        if tools := sss.get("tools"):
            tools = cast(list[BaseTool], tools)
            d = {"Name": [t.name for t in tools], "Description": [t.description for t in tools]}
            st.dataframe(d)
    
    # Get user input
    query, submitted = display_input_form(select_block, sample_search)

    if submitted and query:
        await handle_agent_execution(placeholder, demo, query)


# Run the async main function only when executing in Streamlit context
try:
    _ = st.session_state  # This will raise an exception if not in Streamlit context
    asyncio.run(main())
except (AttributeError, RuntimeError, Exception):
    # We're being imported, not running in Streamlit - skip execution
    pass
