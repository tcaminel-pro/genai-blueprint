import os
import runpy
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from loguru import logger

from src.ai_core.cache import LlmCache
from src.ai_core.llm import LlmFactory
from src.utils.config_mngr import config_loguru, global_config

os.environ["BLUEPRINT_CONFIG"] = "edc_local"

st.set_page_config(
    page_title="GenAI Lab and Practicum",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(verbose=True)

config_loguru()

logger.info("Start Webapp...")

config = global_config()

logo_eviden = str(Path.cwd() / "src/webapp/static/eviden-logo-white.png")


st.sidebar.success("Select a demo above.")

title_col1, title_col2, title_col3 = st.columns([3, 1, 1])
title_col2.image(logo_eviden, width=120)
# title_col2.image(logo_an, width=120)
title_col1.markdown(
    """
    ## Demos and practicum floor<br>
    **👈 Select one from the sidebar** """,
    unsafe_allow_html=True,
)


def config_sidebar():
    with st.sidebar:
        with st.expander("LLM Configuration", expanded=False):
            current_llm = config.get_str("llm.default_model")
            index = LlmFactory().known_items().index(current_llm)
            llm = st.selectbox("default", LlmFactory().known_items(), index=index, key="select_llm")
            config.set("llm.default_model", str(llm))

            set_debug(
                st.checkbox(
                    label="Debug",
                    value=False,
                    help="LangChain debug mode",
                )
            )
            set_verbose(
                st.checkbox(
                    label="Verbose",
                    value=True,
                    help="LangChain verbose mode",
                )
            )

            LlmCache.set_method(st.selectbox("Cache", ["memory", "sqlite"], index=1))

            if "LUNARY_APP_ID" in os.environ:
                if st.checkbox(label="Use Lunary.ai for monitoring", value=False, disabled=True):
                    config.set("monitoring.default", "lunary")
            if "LANGCHAIN_API_KEY" in os.environ:
                if st.checkbox(label="Use LangSmith for monitoring", value=True):
                    config.set("monitoring.default", "langsmith")
                    os.environ["LANGCHAIN_TRACING_V2"] = "true"
                    os.environ["LANGCHAIN_PROJECT"] = config.get_str("monitoring.project")
                    os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "1.0"

                else:
                    os.environ.pop("LANGCHAIN_TRACING_V2", None)
                    config.set("monitoring.default", "none")


def run_app():
    # taken from https://blog.yericchen.com/python/installable-streamlit-app.html
    # Does not work as expected
    script_path = os.path.abspath(__file__)
    sys.argv = ["streamlit", "run", script_path] + sys.argv[1:]
    runpy.run_module("streamlit", run_name="__main__")
