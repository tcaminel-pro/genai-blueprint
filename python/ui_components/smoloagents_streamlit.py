"""Ported from https://github.com/huggingface/smolagents/blob/main/src/smolagents/gradio_ui.py."""

#!/usr/bin/env python
# coding=utf-8
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

import re
from typing import Optional

import streamlit as st
from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available


def display_step(step_log: MemoryStep):
    """Display agent steps in Streamlit with proper formatting"""
    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        st.markdown(f"**{step_number}**")

        # Display the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            model_output = step_log.model_output.strip()
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
            model_output = model_output.strip()
            st.markdown(model_output)

        # Handle tool calls
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"

            # Display tool call information
            args = first_tool_call.arguments
            content = str(args.get("answer", str(args))) if isinstance(args, dict) else str(args).strip()

            if used_code:
                content = re.sub(r"```.*?\n", "", content)
                content = re.sub(r"\s*<end_code>\s*", "", content)
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            st.markdown(f"🛠️ Used tool {first_tool_call.name}")
            st.markdown(content)

            # Display execution logs
            if hasattr(step_log, "observations") and step_log.observations and step_log.observations.strip():
                log_content = step_log.observations.strip()
                log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                if log_content:
                    st.markdown("📝 Execution Logs")
                    st.markdown(log_content)

            # Display errors
            if hasattr(step_log, "error") and step_log.error is not None:
                st.error(str(step_log.error))

        # Handle standalone errors
        elif hasattr(step_log, "error") and step_log.error is not None:
            st.error(str(step_log.error))

        # Display token and duration information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            step_footnote += (
                f" | Input-tokens:{step_log.input_token_count:,}Output-tokens:{step_log.output_token_count:,}"
            )
        if hasattr(step_log, "duration"):
            step_footnote += f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else ""
        st.caption(step_footnote)
        st.markdown("---")


def stream_to_streamlit(agent, task: str, reset_agent_memory: bool = False, additional_args: Optional[dict] = None):
    """Runs an agent with the given task and streams the messages to Streamlit."""
    if not _is_package_available("streamlit"):
        raise ModuleNotFoundError("Please install 'streamlit' to use the Streamlit UI: `pip install streamlit`")
    total_input_tokens = 0
    total_output_tokens = 0
    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        debug(agent.model.last_input_token_count)
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count"):
            try:
                total_input_tokens += agent.model.last_input_token_count
                total_output_tokens += agent.model.last_output_token_count
                if isinstance(step_log, ActionStep):
                    step_log.input_token_count = agent.model.last_input_token_count
                    step_log.output_token_count = agent.model.last_output_token_count
            except Exception as ex:
                debug(ex)

        display_step(step_log)

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        st.markdown(f"**Final answer:**\n{final_answer.to_string()}\n")
    elif isinstance(final_answer, AgentImage):
        st.image(final_answer.to_raw())
    elif isinstance(final_answer, AgentAudio):
        st.audio(final_answer.to_string(), format="audio/wav")
    else:
        st.markdown(f"**Final answer:** {str(final_answer)}")
