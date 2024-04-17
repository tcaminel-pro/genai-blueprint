"""
Demo of an LLM Augmented Autonomous Agent for Maintenance

Copyright (C) 2023 Eviden. All rights reserved
"""

from datetime import datetime
import sys
import os
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

from devtools import debug
from loguru import logger
import streamlit as st
import pandas as pd
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler


# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on


# Fix issue with ChromaDB that needs recent SQLite version, not available in selected docker base image
# st.session_state.first_time = "first_time" not in st.session_state
# try:
#     __import__("pysqlite3")
#     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# except Exception as ex:
#     if st.session_state.first_time:
#         logger.warning(
#             "Cannot import pysqlite3; ChromaDB might work however. It not, reinstall pysqlite3-binary"
#         )


from python.st_utils.clear_result import with_clear_container
from python.core.maintenance_agents import MaintenanceAgent, PROCEDURES
from python.GenAI_Training import LOGO, app_conf, config_sidebar
from python.core.dummy_data import DATA_PATH, dummy_database
from python.core.coder_agents import DiagramGeneratorTool

# fmt:off
SAMPLE_PROMPTS = {
  #  "Display a diagram" : "Display a diagram",

    "Task prerequisite and required tools": 
        "What are the required tools and prerequisite for task 'Diaphragm Inspection' of procedure 'Power Plant Steam Turbine'?",
    "Required tools availability and localization":  """
            Print the required tools for task 'Diaphragm Inspection' in procedure 'Power Plant Steam Turbine', 
            check if they are available and print their localization""",
    "Tasks assigned to an employee":
            "print the tasks assigned to employee 'John Smith' next week",
    "Print assigned tasks to an employee, spares needed, required tools, their availability and localization": """
        Follow these steps: 
        -  print the tasks assigned to employee John Smith next week. 
        -  print the tools required for these tasks
        -  print the spare parts required for these tasks. 
        -  for each of these spare parts, print their availability and localization.
        Print different sections for each step.""",
    "Values from the sensor 'signal_1' last 2 weeks":
        "Values from the sensor 'signal_1' last week ",
}
# fmt:on


from streamlit.external.langchain.streamlit_callback_handler import LLMThought


original_on_tool_end = LLMThought.on_tool_end
original_on_llm_start = LLMThought.on_llm_start


def new_on_tool_end(
    self,
    output: str,
    color: Optional[str] = None,
    observation_prefix: Optional[str] = None,
    llm_prefix: Optional[str] = None,
    **kwargs: Any,
) -> None:
    original_on_tool_end(self, output, color, observation_prefix, llm_prefix, **kwargs)
    debug(kwargs, observation_prefix, llm_prefix)
    self._container.write("TOOL END")


def new_on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]) -> None:
    original_on_llm_start(self, serialized, prompts)
    debug(serialized, prompts)
    self._container.write(f"LLM_START: {';'.join(prompts)}")


# LLMThought.on_tool_end = new_on_tool_end
# LLMThought.on_llm_start = new_on_llm_start

# fmt=off


if not st.session_state.get("authenticated"):
    st.write("not authenticated")
    st.stop()

config_sidebar()


def agent():
    assert app_conf().embeddings_model

    agent = MaintenanceAgent(
        default_llm=app_conf().chat_gpt,
        embeddings_model=app_conf().embeddings_model,  # type: ignore
    )
    agent.create_tools()
    # agent.add_tools([DiagramGeneratorTool()])
    return agent


################################
#  UI
################################

title_col1, title_col2 = st.columns([2, 1])

title_col1.title("Maintenance Operation Assistant")
title_col2.image(LOGO, width=250)
title_col1.markdown(
    f"""
    ## LLM-Augmented Autonomous Agents (LAAA) Demo for Maintenance Operation.
    

    The goal is to support  plant turbine maintenance operations. 
    Currently, we gather information from:
    - Maintenance procedures  (text document)
    - Planning System   (SQL database)
    - ERP and PLM (API)
""",
    unsafe_allow_html=True,
)

b_column = st.columns(4)
if b_column[0].button("See Planning DB"):
    default_sql = """SELECT * FROM "tasks" ;"""
    with st.expander("Tasks Planning Database", expanded=True):
        sql = st.text_input(label="query", value=default_sql, key="sql_input")
        df = pd.read_sql(sql, dummy_database())
        st.write(df)
        print(sql)

if b_column[1].button("See Sensors Values DB"):
    sql = """SELECT *  FROM "sensor_data" ;"""
    df = pd.read_sql(sql, dummy_database())
    with st.expander("Sensor Values Database", expanded=True):
        st.write(sql)
        st.write(df)

if b_column[2].button("See Procedure", key="procedure"):
    with st.expander("Maintenance Procedure", expanded=True):
        with open(DATA_PATH / PROCEDURES[0], "r") as file:
            st.write(file.read())


def extract_right_part(string: str, separator) -> str:
    return string.split(separator)[1].strip() if separator in string else string.strip()


if b_column[3].button("See tools"):
    with st.expander("Available tools", expanded=True):
        tool_list = [
            (tool.name, extract_right_part(tool.description, "-> str"))
            for tool in agent().tools
        ]
        df = pd.DataFrame(tool_list, columns=["name", "description"])
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "name": st.column_config.Column(width="small"),
                "description": st.column_config.Column(width="large"),
            },  # does  not seems to work :(
        )

sample_prompt_key = st.selectbox(
    "Sample prompts", SAMPLE_PROMPTS.keys(), index=None, placeholder="choose an example"
)
sample_prompt = ""
if sample_prompt_key:
    sample_prompt = dedent(SAMPLE_PROMPTS.get(sample_prompt_key, "").strip())
with st.form(key="form"):
    user_input = st.text_area(
        label="Or, ask your own question / prompt", value=sample_prompt, height=150
    )
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)
    answer_container = output_container.chat_message("assistant", avatar="🛠️")

    context = (
        f"Current date is: {datetime.now().isoformat()}. Use it for SQL the queries."
    )
    query = context + "\n" + user_input

    answer, struct_answer = agent().run(
        query,
        extra_callbacks=[StreamlitCallbackHandler(answer_container)],
        extra_metadata={"st_container": ("answer_container", answer_container)},
    )

    if struct_answer is not None:
        tab1, tab2 = st.tabs(["✉️ Text", "🗃 Structured"])
        tab1.write(answer)
        tab2.write(struct_answer)
    else:
        answer_container.write(answer)
