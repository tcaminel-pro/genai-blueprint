# Source : https://raw.githubusercontent.com/huggingface/smolagents/refs/heads/main/src/smolagents/gradio_ui.py

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
"""
Streamlit UI components for smolagents integration.
Provides functions to display agent steps and stream agent outputs in Streamlit apps.
"""


# 1 - Add a parameter 'display_details' to stream_to_streamlic and _display_step_content. 
# If display_details is False, then do not display the code and the footnote
# 2 - Replace global 'st' by a variable 'container' passed as parameter (type is DeltaGenerator)
# AI!

import re
from typing import Dict, List, Optional

import streamlit as st
from PIL import Image
from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep


def _get_step_footnote(step_log: MemoryStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
        token_str = f" | Input tokens:{step_log.input_token_count:,} | Output tokens: {step_log.output_token_count:,}"
        step_footnote += token_str
    if hasattr(step_log, "duration"):
        step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
        step_footnote += step_duration
    return step_footnote


def _display_step_content(step_log: MemoryStep) -> None:
    """Display content from agent steps in Streamlit"""
    if isinstance(step_log, ActionStep):
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else "Step"
        st.markdown(f"**{step_number}**")

        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            model_output = step_log.model_output.strip()
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
            st.markdown(model_output)

        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            args = first_tool_call.arguments
            content = str(args.get("answer", str(args))) if isinstance(args, dict) else str(args).strip()

            if first_tool_call.name == "python_interpreter":
                content = re.sub(r"```.*?\n", "", content)
                content = re.sub(r"\s*<end_code>\s*", "", content)
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"
                with st.expander(f"🛠️ Used tool {first_tool_call.name}"):
                    st.code(content)

        if hasattr(step_log, "observations") and step_log.observations is not None and step_log.observations.strip():
            log_content = step_log.observations.strip()
            log_content = re.sub(r"^Execution logs:\s*", "", log_content)
            with st.expander("📝 Execution Logs"):
                st.code(log_content, language="bash")

        if hasattr(step_log, "error") and step_log.error is not None:
            st.error(str(step_log.error))

        if getattr(step_log, "observations_images", []):
            for image in step_log.observations_images:
                st.image(AgentImage(image).to_raw())

        st.caption(_get_step_footnote(step_log, step_number))
        st.divider()

    elif isinstance(step_log, PlanningStep):
        st.markdown("**Planning step**")
        st.markdown(step_log.plan)
        st.caption(_get_step_footnote(step_log, "Planning step"))
        st.divider()

    elif isinstance(step_log, FinalAnswerStep):
        final_answer = step_log.final_answer
        if isinstance(final_answer, AgentText):
            st.markdown(f"**Final answer:**\n{final_answer.to_string()}\n")
        elif isinstance(final_answer, AgentImage):
            st.image(final_answer.to_raw())
        elif isinstance(final_answer, AgentAudio):
            st.audio(final_answer.to_raw())
        else:
            st.markdown(f"**Final answer:** {str(final_answer)}")


def stream_to_streamlit(
    agent: MultiStepAgent,
    task: str,
    task_images: Optional[List[Image.Image]] = None,
    reset_agent_memory: bool = False,
    additional_args: Optional[Dict] = None,
) -> None:
    """Runs an agent with the given task and streams the messages to Streamlit components.

    Args:
        agent: The MultiStepAgent instance to run
        task: The task prompt for the agent
        task_images: Optional list of PIL images to include with the task
        reset_agent_memory: Whether to reset the agent's memory before running
        additional_args: Additional arguments to pass to the agent's run method
    """
    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(
        task, images=task_images, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        if getattr(agent.model, "last_input_token_count", None) is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, (ActionStep, PlanningStep)):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        _display_step_content(step_log)
