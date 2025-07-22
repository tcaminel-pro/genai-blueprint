"""
Streamlit page for editing demo YAML configuration files dynamically.

This module provides a web interface to view and edit YAML configuration files
used for various demos. It includes real-time syntax validation, change tracking,
and safe file saving capabilities.
"""

from pathlib import Path
from typing import List

import streamlit as st
import yaml
from devtools import debug  # noqa: F401
from loguru import logger
from pydantic import BaseModel
from streamlit_monaco import st_monaco

from src.utils.config_mngr import OmegaConfig


class DemoConfigEditor(BaseModel):
    """Streamlit page for editing demo YAML configuration files dynamically."""

    @staticmethod
    def list_demo_yaml_files() -> List[Path]:
        """List all YAML files in the config/demos directory."""
        demos_dir = Path("config/demos")
        if not demos_dir.exists():
            return []
        return list(demos_dir.glob("*.yaml"))

    @staticmethod
    def load_yaml_file(file_path: Path) -> str:
        """Load YAML file content as raw text."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            st.error(f"Error loading YAML file: {e}")
            with st.expander("Show full error details"):
                st.exception(e)
            return ""

    @staticmethod
    def save_yaml_file(file_path: Path, content: str) -> bool:
        """Save YAML content back to file.

        Args:
            file_path: Path to save the YAML file
            content: Raw YAML string content to save

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First validate the YAML is valid before saving
            yaml.safe_load(content)

            with open(file_path, "w", encoding="utf-8") as f:
                logger.info(f"Write file : {file_path}")
                f.write(content)
            return True
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML syntax: {e}")
            return False
        except Exception as e:
            st.error(f"Error saving YAML file: {e}")
            return False

    @staticmethod
    def main() -> None:
        """Main page rendering with interactive YAML editor."""
        st.set_page_config(page_title="Demo Config Editor", page_icon="⚙️", layout="wide")

        st.title("🛠️ Demo Configuration Editor")
        st.markdown("Edit demo YAML configuration files dynamically")

        # Get available YAML files
        yaml_files = DemoConfigEditor.list_demo_yaml_files()

        if not yaml_files:
            st.error("No YAML files found in config/demos directory")
            return

        # File selection
        selected_file = st.sidebar.selectbox(
            "Select YAML file to edit", options=yaml_files, format_func=lambda x: x.name
        )

        # Load raw YAML content
        yaml_content = DemoConfigEditor.load_yaml_file(selected_file)
        if not yaml_content:
            st.error("Failed to load configuration file")
            return

        # Editor in a form to prevent partial reruns
        with st.form(key="yaml_editor_form"):
            st.header("YAML Code Editor")
            st.info("Edit the YAML directly with syntax highlighting and validation")

            current_file_key = f"editor_content_{selected_file.name}"
            
            if current_file_key in st.session_state:
                editor_content = st.session_state[current_file_key]
            else:
                editor_content = yaml_content
                st.session_state[current_file_key] = editor_content

            edited_text = st_monaco(
                value=editor_content, height="400px", language="yaml", theme="vs-dark", minimap=False, lineNumbers=True
            )

            if edited_text and edited_text.strip():
                st.session_state[current_file_key] = edited_text
                st.session_state.file_changed = edited_text.strip() != yaml_content.strip()

            # Form submit buttons
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.form_submit_button("💾 Save", use_container_width=True):
                    edited_content = st.session_state.get(current_file_key, yaml_content)
                    try:
                        yaml.safe_load(edited_content)
                        if DemoConfigEditor.save_yaml_file(selected_file, edited_content):
                            st.success("✅ Configuration saved successfully!")
                            st.session_state.file_changed = False
                            st.session_state[current_file_key] = edited_content
                        else:
                            st.error("❌ Failed to save configuration")
                    except yaml.YAMLError as e:
                        st.error(f"Invalid YAML syntax: {e}")
                    except Exception as e:
                        st.error(f"Error saving configuration: {e}")
            
            with col2:
                if st.form_submit_button("❌ Cancel", use_container_width=True):
                    # Reset to original content
                    st.session_state[current_file_key] = yaml_content
                    st.session_state.file_changed = False
                    st.rerun()

        # Status and file info
        st.sidebar.markdown("---")

        # Show change status
        has_changes = st.session_state.get("file_changed", False)
        if has_changes:
            st.sidebar.warning("⚠️ Unsaved changes detected")
        else:
            st.sidebar.success("✅ No unsaved changes")

        st.sidebar.info(f"**File Path:**\n`{selected_file}`\n\n**File Size:** {selected_file.stat().st_size:,} bytes")
        OmegaConfig.singleton.invalidate()  # type: ignore


if __name__ == "__main__":
    DemoConfigEditor.main()
