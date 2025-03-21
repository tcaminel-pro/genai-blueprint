from pathlib import Path

import streamlit as st
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    VisitWebpageTool,
)

from src.ai_core.llm import LlmFactory
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.smoloagents_streamlit import stream_to_streamlit

# MODEL_ID = "gpt_4o_azure"
MODEL_ID = None


# SmolagentsInstrumentor().instrument(tracer_provider=get_telemetry_trace_provider())

SAMPLE_PROMPTS = [
    "How many seconds would it take for a leopard at full speed to run through Pont des Arts?",
    "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?",
    "Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men’s FIFA World Cup?",
]

SAMPLE_PROMPTS = ["What is the weather in San Francisco ? ", "What's it known for?"]


st.title("SmolAgents Chat")
st.logo(str(Path.cwd() / "src/webapp/static/eviden-logo-white.png"), size="large")
llm_config_widget(st.sidebar)


model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)
debug(model_name)


with st.expander(label="Prompt examples", expanded=True):
    st.write(SAMPLE_PROMPTS)


# Input for new messages
if prompt := st.chat_input("What would you like to ask SmolAgents?"):
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=llm)
    with st.container(height=600):
        stream_to_streamlit(agent, prompt)
