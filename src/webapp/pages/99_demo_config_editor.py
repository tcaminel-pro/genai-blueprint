from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml
from code_editor import code_editor
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel


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
    def load_yaml_file(file_path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file using OmegaConf."""
        try:
            # Load the file directly with OmegaConf
            config = OmegaConf.load(str(file_path))
            # Convert to dict for editing
            return OmegaConf.to_container(config, resolve=True) or {}
        except Exception as e:
            st.error(f"Error loading YAML file: {e}")
            with st.expander("Show full error details"):
                st.exception(e)
            return {}

    @staticmethod
    def save_yaml_file(file_path: Path, data: Dict[str, Any]) -> bool:
        """Save data back to YAML file."""
        try:
            # Save raw edited data directly
            with open(file_path, "w", encoding="utf-8") as f:
                logger.info(f"Write file : {file_path}")
                yaml.safe_dump(st.session_state.edited_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving YAML file: {e}")
            return False

    @staticmethod
    def yaml_to_editor_content(data: Dict[str, Any]) -> str:
        """Convert dict data to formatted YAML string for editor."""
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)

    @staticmethod
    def main():
        """Main page rendering."""
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

        # Load current data
        current_data = DemoConfigEditor.load_yaml_file(selected_file)

        if not current_data:
            st.error("Failed to load configuration file")
            return

        # Use the code editor
        st.header("YAML Code Editor")
        st.info("Edit the YAML directly with syntax highlighting and validation")

        # Initialize editor content - use session state if available and for the same file
        current_file_key = f"editor_content_{selected_file.name}"
        yaml_content = DemoConfigEditor.yaml_to_editor_content(current_data)

        # Use saved editor content if it exists for this file, otherwise use file content
        if current_file_key in st.session_state:
            editor_content = st.session_state[current_file_key]
        else:
            editor_content = yaml_content
            st.session_state[current_file_key] = editor_content

        editor_response = code_editor(
            editor_content,
            lang="yaml",
            height="400px",
            theme="dark",
            options={
                "wrap": True,
                "fontSize": 14,
                "minimap": {"enabled": False},
                "automaticLayout": True,
            },
        )

        # Parse edited data - use session state to preserve changes
        try:
            if editor_response["text"]:
                # Save the current editor content for this file
                st.session_state[current_file_key] = editor_response["text"]

                edited_data = yaml.safe_load(editor_response["text"])
                # Always update session state with the current editor content
                st.session_state.edited_data = edited_data

                # Check if content has changed from the original file
                if editor_response["text"].strip() != yaml_content.strip():
                    st.success("YAML parsed successfully! Changes detected.")
                    st.session_state.file_changed = True
                else:
                    st.session_state.file_changed = False
            else:
                edited_data = current_data
                st.session_state.edited_data = edited_data
                st.session_state.file_changed = False
        except yaml.YAMLError as e:
            st.error(f"YAML parsing error: {e}")
            edited_data = current_data
            st.session_state.edited_data = edited_data
            st.session_state.file_changed = False

        # Save button
        st.sidebar.markdown("---")
        save_col, reload_col = st.sidebar.columns(2)

        with save_col:
            # Show if there are unsaved changes
            has_changes = st.session_state.get("file_changed", False)
            button_text = "💾 Save Changes" if has_changes else "💾 Save"
            button_type = "primary" if has_changes else "secondary"

            if st.button(button_text, type=button_type, use_container_width=True):
                # Use the current edited data from session state
                data_to_save = st.session_state.get("edited_data", current_data)
                success = DemoConfigEditor.save_yaml_file(selected_file, data_to_save)
                if success:
                    st.success("✅ Configuration saved successfully!")
                    st.session_state.file_changed = False
                    # Clear cache and reload fresh data from file
                    if current_file_key in st.session_state:
                        del st.session_state[current_file_key]
                    # Reload the saved data to ensure consistency
                    current_data = DemoConfigEditor.load_yaml_file(selected_file)
                    st.session_state.edited_data = current_data
                    st.balloons()
                else:
                    st.error("❌ Failed to save configuration")

        with reload_col:
            if st.button("🔄 Reload", use_container_width=True):
                # Clear the editor content cache for this file
                if current_file_key in st.session_state:
                    del st.session_state[current_file_key]
                st.rerun()

        # Reload button
        if st.sidebar.button("🔄 Reload from File", use_container_width=True):
            # Clear the editor content cache for this file
            if current_file_key in st.session_state:
                del st.session_state[current_file_key]
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


if __name__ == "__main__":
    DemoConfigEditor.main()
