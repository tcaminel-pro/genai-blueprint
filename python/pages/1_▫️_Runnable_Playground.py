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
from python.config import get_config_str
from python.GenAI_Lab import config_sidebar

st.set_page_config(layout="wide")
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

first_example = runnable_desc.examples[0]

if diagram := runnable_desc.diagram:
    file = Path.cwd() / diagram
    st.image(str(file))
    st.write("")

if path := first_example.path:
    sel_col1, sel_col2 = st.columns(2)
    uploaded_file = sel_col1.file_uploader(
        "Upload a text file",
        accept_multiple_files=False,
        type=["*.txt"],
    )
    sel_col2.write("Or else use:")
    default_file_name = sel_col2.radio(
        "", options=[first_example.path], index=None, horizontal=True
    )
    if uploaded_file:
        path = Path(uploaded_file.name)

llm_id = get_config_str("llm", "default_model")
config = {}
first_example = runnable_desc.examples[0]
config |= {"llm": llm_id}
if path:
    config |= {"path": path}
elif first_example.path:
    config |= {"path": first_example.path}

with st.expander("Runnable Graph", expanded=False):
    runnable = runnable_desc.get(config)
    if importlib.util.find_spec("pygraphviz") is None:
        # st.graphviz_chart
        # drawing = runnable.get_graph().draw_ascii()
        drawing = runnable.get_graph().draw_mermaid_png()
    else:
        drawing = runnable.get_graph().draw_png()  # type: ignore
    st.write(drawing)

# selected_runnable = st.selectbox("Select a Runnable", list(RUNNABLES.keys()))


with st.form("my_form"):
    input = st.text_area("Enter input:", first_example.query[0], placeholder="")
    submitted = st.form_submit_button("Submit")
    if submitted:
        if get_config_str("monitoring", "default") == "langsmith":
            # use Langsmith context manager to get the UTL to the trace
            with tracing_v2_enabled() as cb:
                result = runnable_desc.invoke(input, config)
                url = cb.get_run_url()
                st.write("[trace](%s)" % url)
        else:
            result = runnable_desc.invoke(input, config)
        st.write(result)
