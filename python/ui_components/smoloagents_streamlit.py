"""Ported from https://github.com/huggingface/smolagents/blob/main/src/smolagents/gradio_ui.py

Aider prompt :
    Port following code from Gradio to Streamlit and replace it.
    Modify class and function name, but don't add new class method.
    Use the 'to_raw' method to use AgentImage in st.image().
    Replace relative import by absolute import from smolagents.
    Add in comment that it's a port from  https://github.com/huggingface/smolagents/blob/main/src/smolagents/gradio_ui.py.
    Insert that prompt in the module comments.


"""

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
import mimetypes
import os
import re
from typing import Optional

from smolagents.agents import ActionStep, AgentStepLog, MultiStepAgent
from smolagents.types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.utils import _is_package_available


def display_step(step_log: AgentStepLog):
    """Display agent steps in Streamlit"""
    import streamlit as st

    if isinstance(step_log, ActionStep):
        st.write(step_log.llm_output or "")
        if step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "code interpreter"
            content = first_tool_call.arguments
            if used_code:
                content = f"```py\n{content}\n```"
            st.markdown(f"**🛠️ Used tool {first_tool_call.name}**")
            st.code(content)
        if step_log.observations is not None:
            st.write(step_log.observations)
        if step_log.error is not None:
            st.error(str(step_log.error))


def stream_to_streamlit(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and displays the messages in Streamlit."""
    if not _is_package_available("streamlit"):
        raise ModuleNotFoundError("Please install 'streamlit' to use the StreamlitUI: `pip install streamlit`")
    import streamlit as st

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        display_step(step_log)

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        st.markdown(f"**Final answer:**\n{final_answer.to_string()}\n")
    elif isinstance(final_answer, AgentImage):
        st.image(final_answer.to_raw())
    elif isinstance(final_answer, AgentAudio):
        st.audio(final_answer.to_string())
    else:
        st.write(str(final_answer))


class StreamlitUI:
    """A one-line interface to launch your agent in Streamlit"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("streamlit"):
            raise ModuleNotFoundError("Please install 'streamlit' to use the StreamlitUI: `pip install streamlit`")
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def upload_file(
        self,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import streamlit as st

        uploaded_file = st.file_uploader(
            "Upload a file", type=[mimetypes.guess_extension(t) for t in allowed_file_types]
        )
        if uploaded_file is None:
            return None

        try:
            mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        except Exception as e:
            st.error(f"Error: {e}")
            return None

        if mime_type not in allowed_file_types:
            st.error("File type disallowed")
            return None

        # Sanitize file name
        original_name = os.path.basename(uploaded_file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File uploaded: {file_path}")
        return file_path

    def launch(self):
        import streamlit as st

        st.title("Smol Agents")

        if self.file_upload_folder is not None:
            file_path = self.upload_file()
            if file_path:
                st.session_state.file_uploads = st.session_state.get("file_uploads", []) + [file_path]

        prompt = st.text_input("Enter your message:")
        if prompt:
            if st.session_state.get("file_uploads"):
                prompt += f"\nYou have been provided with these files, which might be helpful or not: {st.session_state.file_uploads}"

            stream_to_streamlit(self.agent, task=prompt, reset_agent_memory=False)


__all__ = ["stream_to_streamlit", "StreamlitUI"]
