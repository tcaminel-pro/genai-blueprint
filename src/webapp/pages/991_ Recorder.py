import streamlit as st

from src.utils.streamlit.recorder import StreamlitRecorder

str = StreamlitRecorder()

with st.expander("test", expanded=True):
    str.set_container(st.container())
    if st.button("play"):
        with str:
            st.write("Hello")
            st.markdown("World")


if st.button("replay"):
    # Later...
    str.replay()  # Replays at normal speed
    str.replay(speed=2.0)  # Replays at 2x speed
