# Main Streamlit application configuration and setup
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.utils.basic_auth import authenticate, is_authenticated, load_auth_config
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
pages_dir = config().get_dir_path("ui.pages_dir")

# Get navigation structure from config
nav_config = config().get("ui.navigation", {})

# Build pages dictionary with sections
pages = {}

if nav_config:
    # Use explicit navigation structure from config
    for section_name, page_files in nav_config.items():
        section_pages = []
        for page_file in page_files:
            page_path = pages_dir / page_file
            if page_path.exists():
                section_pages.append(st.Page(page_path.absolute()))
        if section_pages:
            pages[section_name.title()] = section_pages
else:
    # Fallback: Sort files by the number at the beginning of their name
    pages_fn = sorted(
        pages_dir.glob("*.py"), key=lambda f: int(f.name.split("_")[0]) if f.name.split("_")[0].isdigit() else 0
    )
    pages = [st.Page(f.absolute()) for f in pages_fn if f.name != "__init__.py"]

pg = st.navigation(pages, position="top")
pg.run()
