# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_pandas_df.py


import os, sys
from pathlib import Path

from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
import streamlit as st
import pandas as pd
from streamlit.runtime.uploaded_file_manager import DeletedFile, UploadedFile
from devtools import debug


# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore 
# fmt: on
from python.core.dummy_data import DATA_PATH
from python.GenAI_Training import EVIDEN_LOGO, config_sidebar, app_conf
from python.st_utils.load_data import FILE_FORMATS, load_data


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(show_spinner=True)
def get_dataframe(file_or_filename: Path | UploadedFile) -> pd.DataFrame | None:
    return load_data(file_or_filename)


SAMPLE_PROMPTS = [
    "What is the proportion of female passengers that survived?",
    "Were there any notable individuals or families aboard ",
    "Plot in a bar chat the proportion of male and female survivors",
    "What was the survival rate of passengers on the Titanic?",
    "Did the passenger class have an impact on survival rates?",
    "What were the ticket fares and cabin locations for the passengers?"
    "What are the demographics (age, gender, etc.) of the passengers on the Titanic?",
]

if not st.session_state.get("authenticated"):
    st.write("not authenticated")
    st.stop()

config_sidebar()
# st.set_page_config(page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜")
title_col1, title_col2 = st.columns([2, 1])

title_col1.title("DataFrame Agent")
title_col2.image(EVIDEN_LOGO, width=250)
title_col1.markdown(
    f"""
    ## Demo of an Agent to analyse a Dataframe.

    That agent could be part of a multi-agent system.
""",
    unsafe_allow_html=True,
)

sel_col1, sel_col2 = st.columns(2)
uploaded_file = sel_col1.file_uploader(
    "Upload a Data file",
    type=list(FILE_FORMATS.keys()),
    on_change=clear_submit,
)
sel_col2.write("Or else use:")
default_file_name = sel_col2.radio(
    "", options=["titanic.csv"], index=None, horizontal=True
)
df: pd.DataFrame | None = None
if uploaded_file:
    df = get_dataframe(uploaded_file)
elif default_file_name:
    df = get_dataframe(DATA_PATH / default_file_name)
if df is not None:
    with st.expander(label="Loaded Dataframe", expanded=False):
        st.dataframe(df)

sample_prompt = None
if default_file_name == "titanic.csv":
    with st.expander(label="Prompt examples", expanded=False):
        text = "".join([f"\n- {s}" for s in SAMPLE_PROMPTS])
        st.markdown(text)


if df is None:
    st.warning(
        "This app uses the  Python interpreter, which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
    )

if "messages" not in st.session_state or st.sidebar.button(
    "Clear conversation history"
):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

sample_prompt = sample_prompt or ""

if prompt := st.chat_input(placeholder=sample_prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    pandas_df_agent = create_pandas_dataframe_agent(
        app_conf().chat_gpt,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS
        if app_conf().use_functions
        else AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(
            st.session_state.messages,
            callbacks=app_conf().callback_handlers + [st_cb],
            metadata={"agentName": "Dataframe Agent"},
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
