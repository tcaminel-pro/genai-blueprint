import asyncio
import uuid
from typing import Literal, Tuple, cast

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import tracing_v2_enabled
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from src.ai_core.llm import get_llm
from src.ai_core.prompts import dedent_ws, dict_input_message
from src.utils.streamlit.thread_issue_fix import get_streamlit_cb
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.streamlit_chat import StreamlitStatusCallbackHandler, display_messages

load_dotenv()

llm_config_widget(st.sidebar, False)

st.title("EcoDesign Chat")


@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


# tools = [ref_product_search_tool, get_weather, get_material_breakdown]
tools = [get_weather]

examples = ["What is the weather in San Francisco ? ", "What's it known for?"]
examples += ["List of reference products matching PMMA"]

SYSTEM_PROMPT = dedent_ws(
    """
    Your are a LCA expert assistant. Use provided tools to answer questions. \n
    - If the user asks for a list of something and that the tool returns a list, print it as Markdown table. 
"""
)

with st.container(border=True):
    col1, col2 = st.columns([3, 2])
    col1.write("Examples")
    col1.write(examples)
    system_prompt_input = col2.text_area("System Prompt", value=SYSTEM_PROMPT, height=200)


@st.cache_resource()
def create_agent() -> Tuple[CompiledGraph, RunnableConfig]:
    llm = get_llm()
    checkpointer = MemorySaver()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt_input or None, checkpointer=checkpointer)
    return agent, cast(RunnableConfig, config)


async def main() -> None:
    display_messages(st)
    agent, config = create_agent()

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
