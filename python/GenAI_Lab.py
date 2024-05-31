import os
from pathlib import Path

import streamlit as st
from langchain.globals import set_debug, set_verbose
from loguru import logger

from python.ai_core.llm import LlmFactory, set_cache
from python.config import get_config, set_config

logger.info("Start Webapp...")

logo_an = str(Path.cwd() / "static" / "AcademieNumerique_Colour_RGB-150x150.jpg")
logo_eviden = str(Path.cwd() / "static/eviden-logo-white.png")


st.set_page_config(
    page_title="GenAI Lab and Practicum",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.success("Select a demo above.")

title_col1, title_col2, title_col3 = st.columns([3, 1, 1])
title_col2.image(logo_eviden, width=120)
title_col2.image(logo_an, width=120)
title_col1.markdown(
    """
    ## Demos and practicum floor<br>
    **👈 Select one from the sidebar** """,
    unsafe_allow_html=True,
)


def config_sidebar():
    with st.sidebar:
        with st.expander("LLM Configuration", expanded=True):
            llm = st.selectbox("default", LlmFactory().known_items(), index=0)
            set_config("llm", "default_model", str(llm))

            set_debug(
                st.checkbox(
                    label="Debug",
                    value=True,
                    help="LangChain debug mode",
                )
            )
            set_verbose(
                st.checkbox(
                    label="Verbose",
                    value=False,
                    help="LangChain verbose mode",
                )
            )

            set_cache(st.selectbox("Cache", ["memory", "sqlite"], index=1))

            if "LUNARY_APP_ID" in os.environ:
                if st.checkbox(label="Use Lunary.ai for monitoring", value=False):
                    set_config("monitoring", "default", "lunary")
            if "LANGCHAIN_API_KEY" in os.environ:
                if st.checkbox(label="Use LangSmith for monitoring", value=True):
                    set_config("monitoring", "default", "langsmith")
                    os.environ["LANGCHAIN_TRACING_V2"] = "true"
                    os.environ["LANGCHAIN_PROJECT"] = get_config(
                        "monitoring", "project"
                    )
                    os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "1.0"

                else:
                    os.environ["LANGCHAIN_TRACING_V2"] = "false"


config_sidebar()
