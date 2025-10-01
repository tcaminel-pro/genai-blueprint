"""Runnable Playground - Interactive testing environment for LangChain Runnables.

This module provides a Streamlit-based interface for testing and exploring LangChain Runnable components.
It allows users to:
- Select from available Runnables
- View diagrams and graphs of the Runnable structure
- Upload or select input files
- Execute Runnables with different configurations
- View execution results and traces

The playground integrates with LangSmith for tracing and monitoring when configured.
"""

from pathlib import Path

import streamlit as st

title_col1, title_col2 = st.columns([2, 1])


title_col1.title("Welcome ! ")
LOGO = "New Atos logo white.png"
logo = str(Path.cwd() / "genai_blueprint/webapp/static" / LOGO)
title_col2.image(logo, width=250)
title_col1.markdown(
    """
    ### ☝️ ☝️ Select a demo or a Playground! 
    """
)
