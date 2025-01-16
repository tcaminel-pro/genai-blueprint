"""
GenAI Lab - Streamlit Web Application

This module serves as the main entry point for the GenAI Lab web application.
It provides a user interface for experimenting with and demonstrating various
Generative AI capabilities using Streamlit.

"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

st.set_page_config(
    page_title="GenAI Lab, Practicum and Demos",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(verbose=True)

logger.info("Start Webapp...")

logo_an = str(Path.cwd() / "static" / "AcademieNumerique_Colour_RGB-150x150.jpg")
logo_eviden = str(Path.cwd() / "static/eviden-logo-white.png")
st.logo(logo_eviden, size="large")


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


def config_sidebar() -> None:
    """Configure the sidebar with LLM settings and monitoring options

    This function creates an expandable sidebar section that allows users to:
    - Select the default LLM model
    - Enable/disable debug and verbose modes
    - Configure caching method (memory or sqlite)
    - Enable monitoring through LangSmith or Lunary.ai

    The configuration is persisted using the application's config system.
    """
