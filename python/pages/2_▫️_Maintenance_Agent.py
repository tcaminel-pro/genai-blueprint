"""
Demo of an LLM Augmented Autonomous Agent for Maintenance

Copyright (C) 2023 Eviden. All rights reserved
"""

from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st
from langchain.callbacks import tracing_v2_enabled
from langchain_community.callbacks import StreamlitCallbackHandler
from langsmith import Client
from loguru import logger  # noqa: F401

from python.ai_agents.maintenance_agents import PROCEDURES, MaintenanceAgent
from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.llm import LlmFactory, get_llm
from python.demos.maintenance_agent.maintenance_data import DATA_PATH, dummy_database
from python.GenAI_Lab import config_sidebar
from python.utils.streamlit.clear_result import with_clear_container

# fmt:off
SAMPLE_PROMPTS = {
  #  "Display a diagram" : "Display a diagram",

    "Task prerequisite and required tools": 
        "What are the required tools and prerequisite for task 'Diaphragm Inspection' of procedure 'Power Plant Steam Turbine'?",
    "Required tools availability and localization":  """
            Print the required tools for task 'Diaphragm Inspection' in procedure 'Power Plant Steam Turbine', 
            check if they are available and print their localization""",
    "Tasks assigned to an employee":
            "print the tasks assigned to employee 'John Smith' next 2 days",
    "Print assigned tasks to an employee, spares needed, required tools, their availability and localization": """
        Follow these steps: 
        -  print the tasks assigned to employee John Smith next 2 days. 
        -  print the tools required for these tasks
        -  print the spare parts required for these tasks. 
        -  for each of these spare parts, print their availability and localization.
        Print different sections for each step.""",
    "Values from the sensor 'signal_1' last 2 weeks":
        "Values from the sensor 'signal_1' last week ",
}
# fmt:on

MODEL = "gemini_pro_google"

config_sidebar()


def agent():
    # llm = get_llm()
    # embeddings_model = EmbeddingsFactory().get()

    agent = MaintenanceAgent(
        llm_factory=LlmFactory(), embeddings_factory=EmbeddingsFactory()
    )
    agent.create_tools()
    # agent.add_tools([DiagramGeneratorTool()])
    return agent


################################
#  UI
################################

title_col1, title_col2 = st.columns([2, 1])

logo_eviden = str(Path.cwd() / "static/eviden-logo-white.png")

title_col1.title("Maintenance Operation Assistant")
title_col2.image(logo_eviden, width=250)
title_col1.markdown(
    """
    ## LLM-Augmented Autonomous Agents Demo for Maintenance Operation.
    

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

    llm = get_llm()
    client = Client()

    with tracing_v2_enabled() as cb:
        answer = agent().run(
            query,
            llm,
            extra_callbacks=[StreamlitCallbackHandler(answer_container)],
            extra_metadata={"st_container": ("answer_container", answer_container)},
        )
        url = cb.get_run_url()
        answer_container.write(answer)
        answer_container.write("[trace](%s)" % url)
