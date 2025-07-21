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
            # Convert dict to OmegaConf and save as YAML
            config = OmegaConf.create(data)
            with open(file_path, "w", encoding="utf-8") as f:
                logger.info(f"Write file : {file_path}")
                OmegaConf.save(config, f)
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

        yaml_content = DemoConfigEditor.yaml_to_editor_content(current_data)
        editor_response = code_editor(
            yaml_content,
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
                edited_data = yaml.safe_load(editor_response["text"])
                if editor_response["text"] != yaml_content:
                    st.success("YAML parsed successfully!")
                    st.session_state.file_changed = True
            else:
                edited_data = current_data
        except yaml.YAMLError as e:
            st.error(f"YAML parsing error: {e}")
            edited_data = current_data

        st.session_state.edited_data = edited_data

        # Save button
        st.sidebar.markdown("---")
        save_col, reload_col = st.sidebar.columns(2)

        with save_col:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                success = DemoConfigEditor.save_yaml_file(selected_file, st.session_state.edited_data)
                if success:
                    st.success("✅ Configuration saved successfully!")
                    st.session_state.file_changed = False
                    st.balloons()
                else:
                    st.error("❌ Failed to save configuration")

        with reload_col:
            if st.button("🔄 Reload", use_container_width=True):
                st.rerun()

        # Reload button
        if st.sidebar.button("🔄 Reload from File", use_container_width=True):
            st.rerun()

        # Backup info
        st.sidebar.markdown("---")
        st.sidebar.info(f"**File Path:**\n`{selected_file}`\n\n**File Size:** {selected_file.stat().st_size:,} bytes")


if __name__ == "__main__":
    DemoConfigEditor.main()
