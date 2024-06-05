# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_pandas_df.py


from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_experimental.agents import create_pandas_dataframe_agent
from loguru import logger  # noqa: F401
from streamlit.runtime.uploaded_file_manager import UploadedFile

from python.ai_core.llm import get_llm
from python.GenAI_Lab import config_sidebar
from python.utils.streamlit.load_data import TABULAR_FILE_FORMATS, load_tabular_data


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(show_spinner=True)
def get_dataframe(file_or_filename: Path | UploadedFile) -> pd.DataFrame | None:
    return load_tabular_data(file_or_filename)


SAMPLE_PROMPTS = [
    "What is the proportion of female passengers that survived?",
    "Were there any notable individuals or families aboard ",
    "Plot in a bar chat the proportion of male and female survivors",
    "What was the survival rate of passengers on the Titanic?",
    "Did the passenger class have an impact on survival rates?",
    "What were the ticket fares and cabin locations for the passengers?"
    "What are the demographics (age, gender, etc.) of the passengers on the Titanic?",
]


st.set_page_config(
    page_title="LangChain: Chat with pandas DataFrame", layout="wide", page_icon="🦜"
)
title_col1, title_col2 = st.columns([2, 1])

config_sidebar()
logo_eviden = str(Path.cwd() / "static/eviden-logo-white.png")

title_col1.title("DataFrame Agent")
title_col2.image(logo_eviden, width=250)
title_col1.markdown(
    """
    ## Demo of an Agent to analyse a Dataframe.

    That agent could be part of a multi-agent system.
""",
    unsafe_allow_html=True,
)

sel_col1, sel_col2 = st.columns(2)
uploaded_file = sel_col1.file_uploader(
    "Upload a Data file",
    type=list(TABULAR_FILE_FORMATS.keys()),
    on_change=clear_submit,
)
sel_col2.write("Or else use:")
default_file_name = sel_col2.radio(
    "", options=["titanic.csv"], index=None, horizontal=True
)

DATA_PATH = Path.cwd() / "use_case_data/other"

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

    llm = get_llm()
    debug(llm)

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="tool-calling",
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(
            st.session_state.messages,
            # callbacks=app_conf().callback_handlers + [st_cb],
            metadata={"agentName": "Dataframe Agent"},
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
