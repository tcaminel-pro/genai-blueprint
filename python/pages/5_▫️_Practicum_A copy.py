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
from python.GenAI_Training import logo, app_conf, config_sidebar
from python.core.dummy_data import DATA_PATH, dummy_database
from python.core.coder_agents import DiagramGeneratorTool

# fmt:off


################################
#  UI
################################

title_col1, title_col2 = st.columns([2, 1])

title_col1.title("Practicum A")
title_col2.image(logo, width=250)
title_col1.markdown(
    f"""
    ##  Your first exercise with a Web App
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
