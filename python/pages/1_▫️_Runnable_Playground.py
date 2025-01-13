"""
Runnable Playground - Interactive testing environment for LangChain Runnables.

This module provides a Streamlit-based interface for testing and exploring LangChain Runnable components.
It allows users to:
- Select from available Runnables
- View diagrams and graphs of the Runnable structure
- Upload or select input files
- Execute Runnables with different configurations
- View execution results and traces

The playground integrates with LangSmith for tracing and monitoring when configured.
"""

import importlib
import importlib.util
from pathlib import Path

import streamlit as st
from langchain.callbacks import tracing_v2_enabled
from pydantic import BaseModel

from python.ai_core.chain_registry import (
    find_runnable,
    get_runnable_registry,
    load_modules_with_chains,
)
from python.GenAI_Lab import config_sidebar

st.title("💬 Runnable Playground")

# Main Streamlit components and their purposes:
# 1. Title: Displays the page title and icon
# 2. Runnable Selector: Allows user to choose which Runnable to test
# 3. Sidebar: Provides configuration options (handled by config_sidebar)
# 4. Diagram Display: Shows visual representation of the Runnable if available
# 5. File Uploader: Allows input file selection for file-based Runnables
# 6. Graph Visualization: Shows the execution graph of the selected Runnable
# 7. Input Form: Provides text input and submission controls

# Load all available runnable components from registered modules
load_modules_with_chains()

# Get list of all available runnables with their tags and names

runnables_list = sorted([(o.tag or "", o.name) for o in get_runnable_registry()])
selection = st.selectbox("Runnable", runnables_list, index=0, format_func=lambda x: f"[{x[0]}] {x[1]}")
if not selection:
    st.stop()
runnable_desc = find_runnable(selection[1])
if not runnable_desc:
    st.stop()

# Configure and display the sidebar with settings and options
# This includes model selection, configuration parameters, and other controls
config_sidebar()

# Get the first example from the runnable description to use as default values
first_example = runnable_desc.examples[0]

if diagram := runnable_desc.diagram:
    file = Path.cwd() / diagram
    st.image(str(file))

if path := first_example.path:
    sel_col1, sel_col2 = st.columns(2)
    uploaded_file = sel_col1.file_uploader(
        "Upload a text file",
        accept_multiple_files=False,
        type=["*.txt"],
    )
    sel_col2.write("Or else use:")
    default_file_name = sel_col2.radio("", options=[first_example.path], index=None, horizontal=True)
    if uploaded_file:
        path = Path(uploaded_file.name)

llm_id = global_config().get_str("llm", "default_model")
config = {}
first_example = runnable_desc.examples[0]
config |= {"llm": llm_id}
if path:
    config |= {"path": path}
elif first_example.path:
    config |= {"path": first_example.path}

with st.expander("Runnable Graph", expanded=False):
    graph = runnable_desc.get(config).get_graph()
    try:
        if importlib.util.find_spec("pygraphviz") is None:
            # st.graphviz_chart
            st.image(graph.draw_mermaid_png())
        else:
            st.image(graph.draw_png())  # type: ignore
    except Exception:
        try:
            st.write(graph.draw_ascii())
        except Exception as ex:
            st.write(f"cannot draw graph: {ex}")


# selected_runnable = st.selectbox("Select a Runnable", list(RUNNABLES.keys()))


with st.form("my_form"):
    input = st.text_area("Enter input:", first_example.query[0], placeholder="")
    submitted = st.form_submit_button("Submit")
    if submitted:
        if global_config().get_str("monitoring", "default") == "langsmith":
            # use Langsmith context manager to get the UTL to the trace
            with tracing_v2_enabled() as cb:
                result = runnable_desc.invoke(input, config)
                url = cb.get_run_url()
                st.write("[trace](%s)" % url)
        else:
            result = runnable_desc.invoke(input, config)
        if isinstance(result, BaseModel):
            st.json(result.json(exclude_none=True))
        else:
            st.write(result)
