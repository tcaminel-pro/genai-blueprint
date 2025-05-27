# Main Streamlit application configuration and setup
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.utils.config_mngr import global_config
from src.utils.logger_factory import setup_logging

load_dotenv(verbose=True)

setup_logging()
# print("Starting Web Application...")


@st.cache_resource()
def config():
    return global_config()


# Configure Streamlit page settings
st.set_page_config(
    page_title=global_config().get_str("ui.app_name"),
    page_icon="🛠️",
    layout="wide",  #
    initial_sidebar_state="expanded",
)

LOGO = "New Atos logo white.png"
logo = str(Path.cwd() / "src/webapp/static" / LOGO)
st.logo(logo, size="medium")
# Get Streamlit pages to display from config
pages_dir = config().get_dir_path("ui.pages_dir")
# Sort files by the number at the beginning of their name
pages_fn = sorted(
    pages_dir.glob("*.py"), key=lambda f: int(f.name.split("_")[0]) if f.name.split("_")[0].isdigit() else 0
)
pages = [st.Page(f.absolute()) for f in pages_fn if f.name != "__init__.py"]
pg = st.navigation(pages)
pg.run()
