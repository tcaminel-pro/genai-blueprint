import streamlit as st
from python.ai_core.chain_registry import get_runnable_registry

st.title("💬 Runnable playground")

runnables_list = sorted([(o.tag, o.name) for o in get_runnable_registry()])
selection = st.selectbox(
    "Runnable", runnables_list, index=0, format_func=lambda x: f"[{x[0]}] {x[1]}"
)
if not selection:
    st.stop()
runnable_desc = next(
    (x for x in get_runnable_registry() if x.name == selection[1]), None
)
assert runnable_desc

# selected_runnable = st.selectbox("Select a Runnable", list(RUNNABLES.keys()))

with st.form("my_form"):
    input = st.text_area("Enter input:", runnable_desc.examples[0], placeholder="")
    submitted = st.form_submit_button("Submit")
    if submitted:
        result = runnable_desc.runnable.invoke({"input": input})
        st.info(result)
