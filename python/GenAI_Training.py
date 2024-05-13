from pathlib import Path

import streamlit as st
from langchain.globals import set_debug, set_verbose

from python.ai_core.llm import set_cache
from python.config import set_config

logo_an = str(Path.cwd() / "static" / "AcademieNumerique_Colour_RGB-150x150.jpg")


st.set_page_config(
    page_title="GenAI Practicum",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.success("Select a demo above.")

title_col1, title_col2 = st.columns([2, 1])
title_col2.image(logo_an, width=120)
title_col1.markdown(
    """
    ## Demos and exercise<br>
    **👈 Select one from the sidebar** """,
    unsafe_allow_html=True,
)


def config_sidebar():
    with st.sidebar:
        with st.expander("LLM Configuration", expanded=True):
            llm = st.selectbox("default", KNOWN_LLM, index=0)
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

            if st.checkbox(
                label="Use Lunary.ai for monitoring",
                value=True,
                help="Lunary.ai is a LLM monitoring service. It's free until 1K event/day ",
            ):
                pass


config_sidebar()
