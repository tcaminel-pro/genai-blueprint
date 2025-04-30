import streamlit as st

from src.utils.streamlit.recorder import StreamlitRecorder

str = StreamlitRecorder(st)
if st.button("play"):
    with str:
        st.write("Hello")
        st.markdown("World")
        with st.expander("expander"):
            st.header("In expander")

if st.button("replay"):
    # Later...
    str.replay()  # Replays at normal speed
    str.replay(speed=2.0)  # Replays at 2x speed
