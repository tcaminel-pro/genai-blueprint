import asyncio
import uuid
from typing import Tuple, cast

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import tracing_v2_enabled
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.ai_core.llm import get_llm
from src.ai_core.mcp_client import get_mcp_servers_dict
from src.ai_core.prompts import dedent_ws, dict_input_message
from src.utils.streamlit.thread_issue_fix import get_streamlit_cb
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.streamlit_chat import StreamlitStatusCallbackHandler, display_messages

MCP_SERVERS = ["weather", "playwright", "ppt"]
# MCP_SERVERS = ["weather"]
load_dotenv()

llm_config_widget(st.sidebar, False)

st.title("Chat Playground")


mcp_enabled = st.toggle("MCP", True)
# tools = [ref_product_search_tool, get_weather, get_material_breakdown]
local_tools = []

examples = ["What is the weather in San Francisco ? ", "What's it known for?"]


examples += [
    "What is the content of current directory ? "
    "Connect to atos.net with a browser, find the page with blogs, and get list of recent blog articles",
    "Get weather in Toulouse today. Create a Powerpoint file in that directty about it and usual climate in Toulouse",
]


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
def get_agent_config() -> tuple[RunnableConfig, BaseCheckpointSaver]:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    checkpointer = MemorySaver()
    return cast(RunnableConfig, config), checkpointer


async def main() -> None:
    display_messages(st)
    config, checkpointer = get_agent_config()
    llm = get_llm()

    mcp_servers_params = get_mcp_servers_dict(MCP_SERVERS) if mcp_enabled else {}
    debug(mcp_servers_params)
    client = None
    try:
        client = MultiServerMCPClient(mcp_servers_params)
        rcp_tools = await client.get_tools()
        all_tools = local_tools + rcp_tools
        debug(all_tools)
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
    finally:
        pass


# Run the async main function
asyncio.run(main())
