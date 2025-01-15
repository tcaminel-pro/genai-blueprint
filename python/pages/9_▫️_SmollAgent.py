import streamlit as st
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
)

from python.ai_core.llm import LlmFactory
from python.ai_extra.smoloagents_streamlit import stream_to_streamlit

MODEL_ID = "gpt_4omini_openai"
model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)


SAMPLE_PROMPTS = {"How many seconds would it take for a leopard at full speed to run through Pont des Arts?"}
st.title("SmolAgents Chat Interface")

with st.expander(label="Prompt examples", expanded=True):
    text = "".join([f"\n- {s}" for s in SAMPLE_PROMPTS])
    st.markdown(text)


# Input for new messages
if prompt := st.chat_input("What would you like to ask SmolAgents?"):
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=llm)
    with st.container(height=600):
        stream_to_streamlit(agent, prompt)
