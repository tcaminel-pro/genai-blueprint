import random

import streamlit as st

from genai_blueprint.utils.streamlit.auto_scroll import scroll_to_here


def test_simple() -> None:
    content_container = st.empty()
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.counter = 0

    def add_message():
        st.session_state.counter += 1
        st.session_state.messages.append(f"Message #{st.session_state.counter}")

    if st.button("Add New Message"):
        add_message()
    with content_container:
        msg = "\n".join(st.session_state.messages)
        # for msg in st.session_state.messages:
        with st.container(height=200):
            st.text(msg)
            scroll_to_here()


def test2() -> None:
    def content():
        for i in range(50):
            st.write(f"line {i}: {random.randint(0, 1000)}")

    st.subheader("Content")
    if st.button("Scroll to bottom"):
        content()
        scroll_to_here()
    else:
        content()


test_simple()

# test2()
