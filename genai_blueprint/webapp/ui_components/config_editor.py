"""Reusable configuration editor dialog for Streamlit applications.

This module provides a standardized configuration editor dialog that can be
used across different Streamlit pages to edit YAML configuration files.
"""

from pathlib import Path

import streamlit as st
import yaml
from streamlit_monaco import st_monaco


@st.dialog("Edit Configuration", width="large")
def edit_config_dialog(config_path: str | Path) -> None:
    """Open a dialog to edit a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file
    """
    config_path = Path(config_path)

    if not config_path.exists():
        st.error(f"Configuration file not found: {config_path}")
        return

    try:
        # Load current configuration
        with open(config_path, "r", encoding="utf-8") as f:
            current_content = f.read()

        # YAML editor
        edited_content = st_monaco(
            value=current_content, height="400px", language="yaml", theme="vs-dark", minimap=False, lineNumbers=True
        )

        col1, col2, col3 = st.columns([1, 1, 2])

        if col1.button("💾 Save", width="stretch"):
            try:
                # Validate YAML
                yaml.safe_load(edited_content)

                # Save file
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(edited_content)

                st.success("Configuration saved successfully!")
                st.info("Reloading page...")
                st.rerun()

            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")
            except Exception as e:
                st.error(f"Error saving file: {e}")

        if col2.button("❌ Cancel", width="stretch"):
            st.rerun()

    except Exception as e:
        st.error(f"Error loading configuration: {e}")
