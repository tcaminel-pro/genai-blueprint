import sys
from pathlib import Path
import streamlit as st
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
)


# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore  # fmt: on

from python.GenAI_Training import config_sidebar
from python.ai_chains.lg_rag_example import rag_chain

config_sidebar()

st.title("💬 Runnable playground")

RUNNABLES: dict[str, Runnable] = {
    "multiply by 2": RunnableLambda(lambda x : float(x)*2) , 
    "first RAG": rag_chain }

selected_runnable = st.selectbox("Select a Runnable", list(RUNNABLES.keys()))

with st.form('my_form'):
    input = st.text_area('Enter input:', '')
    if selected_runnable:
        submitted = st.form_submit_button('Submit')
        if submitted:
            runnable = RUNNABLES[selected_runnable]
            result = runnable.invoke(input)
            st.info(result)
