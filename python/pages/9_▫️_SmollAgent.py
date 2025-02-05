from pathlib import Path

import streamlit as st
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    VisitWebpageTool,
)

from python.ai_core.llm import LlmFactory
from python.ui_components.llm_config import llm_config
from python.ui_components.smoloagents_streamlit import stream_to_streamlit

# MODEL_ID = "gpt_4o_azure"
MODEL_ID = None


# SmolagentsInstrumentor().instrument(tracer_provider=get_telemetry_trace_provider())

SAMPLE_PROMPTS = {
    "How many seconds would it take for a leopard at full speed to run through Pont des Arts?",
    "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?",
    "Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men’s FIFA World Cup?",
}


st.title("SmolAgents Chat")
st.logo(str(Path.cwd() / "static/eviden-logo-white.png"), size="large")
with st.sidebar:
    llm_config()


model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)
debug(model_name)


with st.expander(label="Prompt examples", expanded=True):
    text = "".join([f"\n- {s}" for s in SAMPLE_PROMPTS])
    st.markdown(text)


# Input for new messages
if prompt := st.chat_input("What would you like to ask SmolAgents?"):
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=llm)
    with st.container(height=600):
        stream_to_streamlit(agent, prompt)
