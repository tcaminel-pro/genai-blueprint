import os

import streamlit as st
from langchain.globals import set_debug, set_verbose
from streamlit.delta_generator import DeltaGenerator

from src.ai_core.cache import LlmCache
from src.ai_core.llm import LlmFactory
from src.utils.config_mngr import global_config


def llm_config_widget(parent_container: DeltaGenerator, expanded: bool = True) -> None:
    with parent_container.expander("LLM Configuration", expanded=expanded):
        current_llm = global_config().get_str("llm.default_model")
        index = LlmFactory().known_items().index(current_llm)
        llm = st.selectbox("default", LlmFactory().known_items(), index=index, key="select_llm")
        global_config().set("llm.default_model", str(llm))
        set_debug(
            st.checkbox(
                label="Debug",
                value=False,
                help="LangChain debug mode",
            )
        )
        set_verbose(
            st.checkbox(
                label="Verbose",
                value=True,
                help="LangChain verbose mode",
            )
        )

        LlmCache.set_method(st.selectbox("Cache", ["memory", "sqlite"], index=1))

        if "LUNARY_APP_ID" in os.environ:
            if st.checkbox(label="Use Lunary.ai for monitoring", value=False, disabled=True):
                global_config().set("monitoring.default", "lunary")
        if "LANGCHAIN_API_KEY" in os.environ:
            if st.checkbox(label="Use LangSmith for monitoring", value=True):
                global_config().set("monitoring.default", "langsmith")
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_PROJECT"] = global_config().get_str("monitoring.project")
                os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "1.0"

            else:
                os.environ["LANGCHAIN_TRACING_V2"] = "false"
                global_config().set("monitoring.default", "none")
