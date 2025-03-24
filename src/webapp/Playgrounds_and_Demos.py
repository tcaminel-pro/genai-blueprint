import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from src.utils.config_mngr import config_loguru, global_config
from src.webapp.ui_components.llm_config import llm_config_widget

os.environ["BLUEPRINT_CONFIG"] = "edc_local"

st.set_page_config(
    page_title="GenAI Lab and Practicum",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(verbose=True)

config_loguru()

logger.info("Start Webapp...")

config = global_config()

logo_eviden = str(Path.cwd() / "src/webapp/static/eviden-logo-white.png")

st.sidebar.success("Select a demo above.")

llm_config_widget(st.sidebar)

title_col1, title_col2, title_col3 = st.columns([3, 1, 1])
title_col2.image(logo_eviden, width=120)
# title_col2.image(logo_an, width=120)
title_col1.markdown(
    """
    ## Demos and practicum floor<br>
    **👈 Select one from the sidebar** """,
    unsafe_allow_html=True,
)


def main() -> None:
    # taken from https://blog.yericchen.com/python/installable-streamlit-app.html
    # Does not work as expected
    script_path = __file__
    # sys.argv = ["streamlit", "run", script_path] + sys.argv[1:]
    # runpy.run_module("streamlit", run_name="__main__")

    # corrct 'app' so its hold the name of the file AI!

    # rprint(app)
    # cli.main_run([script_path])

    import subprocess

    # from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    # ctx = get_script_run_ctx()
    # #Some code##
    # process = Popen(["python", script_path])
    # add_script_run_ctx(process, ctx)

    subprocess.run(["streamlit", "run", script_path])
