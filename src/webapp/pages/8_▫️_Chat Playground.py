import asyncio
import uuid
from typing import Literal, Tuple, cast

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

from src.ai_core.llm import get_llm
from src.ai_core.prompts import dedent_ws, dict_input_message
from src.ai_extra.mcp_client import get_mcp_servers_from_config
from src.utils.streamlit.thread_issue_fix import get_streamlit_cb
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.streamlit_chat import StreamlitStatusCallbackHandler, display_messages

load_dotenv()

llm_config_widget(st.sidebar, False)

try:
    st.set_page_config(layout="wide")
except Exception:
    pass
st.title("Chat Playground")


@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


mcp_enabled = st.toggle("MCP", True)
# tools = [ref_product_search_tool, get_weather, get_material_breakdown]
local_tools = [get_weather]

examples = ["What is the weather in San Francisco ? ", "What's it known for?"]
examples += ["List of reference products matching PMMA"]

examples += ["connect to atos.net, find the page with blogs, and get list of recent blog articles"]

SYSTEM_PROMPT = dedent_ws(
    """
    Your are a LCA expert assistant. Use provided tools to answer questions. \n
    - If the user asks for a list of something and that the tool returns a list, print it as Markdown table. 
"""
)

with st.container(border=True):
    col1, col2 = st.columns([3, 3])
    col1.write("Examples")
    col1.write(examples)
    with col2.expander("System prompt"):
        system_prompt_input = st.text_area("System Prompt", value=SYSTEM_PROMPT, height=200)

    with col2.expander("Tools"):
        if tools := st.session_state.get("tools"):
            tools = cast(list[BaseTool], tools)
            d = {t.name: t.description for t in tools}
            st.dataframe(d)


@st.cache_resource()
async def get_rcp_tool() -> list[BaseTool]:
    d = get_mcp_servers_from_config()
    async with MultiServerMCPClient(d) as client:
        # async with MultiServerMCPClient(test_servers) as client:
        return client.get_tools()


# @st.cache_resource()
# def create_agent(rcp_tools: list = []) -> Tuple[CompiledGraph, RunnableConfig]:
#     llm = get_llm()
#     checkpointer = MemorySaver()
#     thread_id = str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}
#     all_tools = tools + rcp_tools
#     # rprint(all_tools)
#     agent = create_react_agent(
#         model=llm, tools=all_tools, prompt=system_prompt_input or None, checkpointer=checkpointer
#     )
#     return agent, cast(RunnableConfig, config)


@st.cache_resource()
def get_agent_config() -> tuple[RunnableConfig, BaseCheckpointSaver]:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    checkpointer = MemorySaver()
    return cast(RunnableConfig, config), checkpointer


async def main() -> None:
    display_messages(st)
    config, checkpointer = get_agent_config()
    llm = get_llm()

    mcp_servers_params = get_mcp_servers_from_config() if mcp_enabled else {}
    async with MultiServerMCPClient(mcp_servers_params) as client:
        rcp_tools = client.get_tools()
        all_tools = local_tools + rcp_tools
        if "tools" not in st.session_state:
            st.session_state["tools"] = all_tools
            st.rerun()
        agent = create_react_agent(
            model=llm, tools=all_tools, prompt=system_prompt_input or None, checkpointer=checkpointer
        )

        if query := st.chat_input():
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


# Run the async main function
asyncio.run(main())
