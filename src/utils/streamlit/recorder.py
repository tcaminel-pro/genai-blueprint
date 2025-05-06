"""Streamlit Recorder Module

Provides functionality to record and replay Streamlit UI actions.


"""

from __future__ import annotations

import time
from typing import Any, Callable, Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


class StreamlitAction:
    """Represents a single Streamlit action with its arguments and timestamp."""

    def __init__(self, func: Callable, args: Tuple, kwargs: dict, timestamp: float) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.timestamp = timestamp


class StreamlitRecorder:
    """
        Records Streamlit actions and allows replaying them.

        Example usage:
    ```
        str = StreamlitRecorder()
        container = st.container()
        with str:
            with container:
                st.write("Hello")
                st.markdown("World")

        # Replay actions
        str.replay(container)  # Replays at normal speed
        str.replay(container, speed=0.01)  # Replays slower
    """

    def __init__(self) -> None:
        if "streamlit_recorder_actions" not in st.session_state:
            st.session_state.streamlit_recorder_actions = []
        if "streamlit_recorder_last_timestamp" not in st.session_state:
            st.session_state.streamlit_recorder_last_timestamp = None
        self.original_functions = {}

    def __enter__(self) -> StreamlitRecorder:
        """Start recording by wrapping Streamlit functions."""
        self._wrap_streamlit_functions()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Stop recording by restoring original functions."""
        self._unwrap_streamlit_functions()

    def _wrap_streamlit_functions(self) -> None:
        """Wrap Streamlit functions to record their calls."""
        functions_to_wrap = ["write", "markdown", "text", "header", "subheader", "code", "dataframe", "expander"]

        for func_name in functions_to_wrap:
            if hasattr(st, func_name):
                original_func = getattr(st, func_name)
                self.original_functions[func_name] = original_func

                def make_wrapper(f: Callable) -> Callable:
                    def wrapper(*args: Any, **kwargs: Any) -> Any:
                        # Record the action
                        now = time.time()
                        time_delta = (
                            now - st.session_state.streamlit_recorder_last_timestamp
                            if st.session_state.streamlit_recorder_last_timestamp
                            else 0
                        )
                        st.session_state.streamlit_recorder_last_timestamp = now
                        st.session_state.streamlit_recorder_actions.append(StreamlitAction(f, args, kwargs, time_delta))
                        # Execute the original function
                        return f(*args, **kwargs)

                    return wrapper

                setattr(st, func_name, make_wrapper(original_func))

    def _unwrap_streamlit_functions(self) -> None:
        """Restore original Streamlit functions."""
        for func_name, original_func in self.original_functions.items():
            setattr(st, func_name, original_func)

    def replay(self, container: DeltaGenerator, speed: float = 1.0) -> None:
        """Replay recorded Streamlit actions.

        Args:
            speed: Speed multiplier for replay (1.0 = normal speed)
        """
        with container:
            for action in st.session_state.streamlit_recorder_actions:
                if action.timestamp > 0:
                    time.sleep(action.timestamp / speed)
                action.func(*action.args, **action.kwargs)

    def clear(self) -> None:
        st.session_state.streamlit_recorder_actions = []
        st.session_state.streamlit_recorder_last_timestamp = None
        self.original_functions = {}
