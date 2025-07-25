# Main Streamlit application configuration and setup
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from src.utils.basic_auth import authenticate, load_auth_config
from src.utils.config_mngr import global_config
from src.utils.logger_factory import setup_logging

load_dotenv(verbose=True)

setup_logging()
# print("Starting Web Application...")


# Configure Streamlit page settings
st.set_page_config(
    page_title=global_config().get_str("ui.app_name"),
    page_icon="🛠️",
    layout="wide",  #
    initial_sidebar_state="expanded",
)

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Authentication handling
auth_config = load_auth_config()

# Only show login form if authentication is enabled and user is not authenticated
if auth_config.enabled and not st.session_state.authenticated:
    st.title("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Stop execution if not authenticated
    st.stop()

# Only show the main application if authenticated
LOGO = "New Atos logo white.png"
logo = str(Path.cwd() / "src/webapp/static" / LOGO)
st.logo(logo, size="medium")

# Get Streamlit pages to display from config
pages_dir = global_config().get_dir_path("ui.pages_dir")

# Get navigation structure from config
nav_config = global_config().get("ui.navigation", {})

# Build pages dictionary with sections
pages = {}


def file_name_to_page_name(file_name: str) -> str:
    """Convert a file name to a formatted page name.

    Removes the leading number and underscores, converts to title case,
    preserves existing capitalization in acronyms and mixed case words.

    Examples:
        '01_CLI_command.py' -> 'CLI Command'
        '02_reAct_demo.py' -> 'ReAct Demo'
        '03_API_demo.py' -> 'API Demo'
        '04_myTool.py' -> 'MyTool'
    """
    try:
        name_without_number = file_name.split("_", 1)[1]
        name_without_ext = name_without_number.rsplit(".", 1)[0]
        words = name_without_ext.split("_")
        formatted_words = []
        for word in words:
            if any(c.isupper() for c in word[1:]):  # Mixed case (e.g. ReAct)
                formatted_words.append(word[0].upper() + word[1:])
            elif word == word.upper():  # All caps (e.g. API)
                formatted_words.append(word)
            else:
                formatted_words.append(word.capitalize())
        return " ".join(formatted_words)
    except Exception:
        return file_name


for section_name, page_files in nav_config.items():
    section_pages = []
    for page_file_name in page_files:
        page_path = pages_dir / page_file_name
        if page_path.exists():
            section_pages.append(st.Page(page=page_path, title=file_name_to_page_name(page_file_name)))
        else:
            logger.warning(f"page not found: {page_path} ")
    if section_pages:
        pages[section_name.title()] = section_pages

pg = st.navigation(pages, position="top")
pg.run()
