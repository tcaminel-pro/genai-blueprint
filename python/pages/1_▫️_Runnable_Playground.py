import importlib
import importlib.util
from pathlib import Path

import streamlit as st
from langchain.callbacks import tracing_v2_enabled

from python.ai_core.chain_registry import (
    find_runnable,
    get_runnable_registry,
    load_modules_with_chains,
)
from python.config import get_config
from python.GenAI_Lab import config_sidebar

st.title("💬 Runnable Playground")

config_sidebar()

load_modules_with_chains()


runnables_list = sorted([f"'{o.name}'" for o in get_runnable_registry()])

runnables_list = sorted([(o.tag, o.name) for o in get_runnable_registry()])
selection = st.selectbox(
    "Runnable", runnables_list, index=0, format_func=lambda x: f"[{x[0]}] {x[1]}"
)
if not selection:
    st.stop()
runnable_desc = find_runnable(selection[1])
if not runnable_desc:
    st.stop()

runnable = runnable_desc.get()

if diagram := runnable_desc.diagram:
    file = Path.cwd() / diagram
    st.image(str(file))
    st.write("")


with st.expander("Runnable Graph", expanded=False):
    if importlib.util.find_spec("pygraphviz") is None:
        st.warning(
            "cannot draw the Runnable graph because pygraphviz and Graphviz are not installed"
        )
    else:
        drawing = runnable.get_graph().draw_png()  # type: ignore
        st.image(drawing)
        st.write("")

# selected_runnable = st.selectbox("Select a Runnable", list(RUNNABLES.keys()))

with st.form("my_form"):
    input = st.text_area("Enter input:", runnable_desc.examples[0], placeholder="")
    submitted = st.form_submit_button("Submit")
    if submitted:
        llm = get_config("llm", "default_model")
        if not input:
            input = runnable_desc.examples[0]

    with tracing_v2_enabled() as cb:
        result = runnable_desc.invoke(input, {"llm": llm})
        st.write(result)
        url = cb.get_run_url()

        st.write("[trace](%s)" % url)
