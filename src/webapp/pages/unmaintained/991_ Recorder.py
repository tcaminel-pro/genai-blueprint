import streamlit as st

from src.utils.streamlit.recorder import StreamlitRecorder

if st.button("sat hello"):
    st.write("Hello")
button = st.button("play")
str = StreamlitRecorder()
container = st.status("Agents thoughts:", expanded=True)
str.replay(container)
if button:
    with str:
        with container:
            st.write("Hello")
            st.markdown("World")


if st.button("replay"):
    # Later...
    str.replay(container)  # Replays at normal speed
    str.replay(container, speed=0.01)  # Replays at normal speed
