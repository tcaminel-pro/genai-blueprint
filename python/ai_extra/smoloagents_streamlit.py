# Port of SmolAgents Gradio interface to Streamlit
# NOT FULLY TESTED !!
#
# Original source : https://github.com/huggingface/smolagents/blob/v1.2.2/src/smolagents/gradio_ui.py
#
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from smolagents.agents import ActionStep, AgentStep, MultiStepAgent
from smolagents.types import AgentAudio, AgentImage, AgentText, handle_agent_output_types


def display_step_content(step_log: AgentStep, test_mode: bool = True):
    """Display agent step content in Streamlit"""
    if isinstance(step_log, ActionStep):
        if step_log.llm_output:
            st.markdown(step_log.llm_output)

        if step_log.tool_call is not None:
            used_code = step_log.tool_call.name == "code interpreter"
            content = step_log.tool_call.arguments
            if used_code:
                st.code(content, language="python")
            else:
                st.markdown(f"**🛠️ Used tool {step_log.tool_call.name}**")
                st.markdown(content)

        if step_log.observations is not None:
            st.markdown(step_log.observations)

        if step_log.error is not None:
            st.error(str(step_log.error))


def stream_to_streamlit(
    agent,
    task: str,
    test_mode: bool = False,
    reset_agent_memory: bool = False,
    **kwargs,
):
    """Runs an agent with the given task and streams the messages to Streamlit"""

    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": task})

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process agent steps
    with st.chat_message("assistant"):
        step_log = ""
        for step_log in agent.run(task, stream=True, reset=reset_agent_memory, **kwargs):
            display_step_content(step_log, test_mode=test_mode)

        # Handle final answer
        final_answer = step_log  #
        final_answer = handle_agent_output_types(final_answer)

        if isinstance(final_answer, AgentText):
            st.markdown(f"**Final answer:**\n{final_answer.to_string()}\n")
        elif isinstance(final_answer, AgentImage):
            st.image(final_answer.to_string())
        elif isinstance(final_answer, AgentAudio):
            st.audio(final_answer.to_string())
        else:
            st.markdown(str(final_answer))

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": str(final_answer)})


class StreamlitUI:
    """A one-line interface to launch your agent in Streamlit"""

    def __init__(self, agent: MultiStepAgent):
        self.agent = agent

    def launch(self):
        st.title("Agent Chat Interface")

        # Input for new messages
        if prompt := st.chat_input("What would you like to ask the agent?"):
            stream_to_streamlit(self.agent, task=prompt, reset_agent_memory=False)
