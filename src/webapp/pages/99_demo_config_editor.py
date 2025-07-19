import os
from pathlib import Path
from typing import Any, Dict, List, Union

import streamlit as st
import yaml
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.utils.config_mngr import global_config


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
            # Use global_config to load and resolve YAML with OmegaConf
            config = global_config().merge_with(str(file_path))
            # Convert OmegaConf to dict for editing
            return OmegaConf.to_container(config, resolve=True) or {}
        except Exception as e:
            st.error(f"Error loading YAML file: {e}")
            return {}

    @staticmethod
    def save_yaml_file(file_path: Path, data: Dict[str, Any]) -> bool:
        """Save data back to YAML file."""
        try:
            # Convert dict to OmegaConf and save as YAML
            config = OmegaConf.create(data)
            with open(file_path, "w", encoding="utf-8") as f:
                OmegaConf.save(config, f)
            return True
        except Exception as e:
            st.error(f"Error saving YAML file: {e}")
            return False

    @staticmethod
    def render_field(key: str, value: Any, parent_key: str = "") -> Any:
        """Render a form field based on the value type."""
        full_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, bool):
            return st.checkbox(key, value=value, key=full_key)
        elif isinstance(value, int):
            return st.number_input(key, value=value, step=1, key=full_key)
        elif isinstance(value, float):
            return st.number_input(key, value=value, key=full_key)
        elif isinstance(value, str):
            return st.text_input(key, value=value, key=full_key)
        elif isinstance(value, list):
            st.subheader(f"List: {key}")
            if len(value) > 0 and isinstance(value[0], (str, int, float)):
                # Simple list of primitives
                items = []
                for i, item in enumerate(value):
                    item_key = f"{full_key}_{i}"
                    items.append(st.text_input(f"{key}[{i}]", value=str(item), key=item_key))

                # Add new item
                new_item = st.text_input(f"Add new item to {key}", key=f"{full_key}_new")
                if st.button(f"Add to {key}", key=f"{full_key}_add"):
                    items.append(new_item)
                    return items
                return items
            else:
                # Complex list (list of dicts)
                items = []
                for i, item in enumerate(value):
                    st.write(f"--- Item {i + 1} ---")
                    items.append(DemoConfigEditor.render_dict(f"{key}_{i}", item))

                if st.button(f"Add new {key} item", key=f"{full_key}_new_item"):
                    items.append({})
                return items
        elif isinstance(value, dict):
            return DemoConfigEditor.render_dict(key, value)
        else:
            return st.text_input(key, value=str(value), key=full_key)

    @staticmethod
    def render_dict(key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a dictionary with nested fields."""
        result = {}
        st.subheader(key)

        with st.expander(f"Edit {key}", expanded=True):
            for k, v in data.items():
                result[k] = DemoConfigEditor.render_field(k, v, key)

        return result

    @staticmethod
    def render_list_editor(key: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Render a list of dictionaries with full editing capability."""
        st.subheader(f"Edit {key}")

        if not items:
            items = []

        edited_items = []

        for idx, item in enumerate(items):
            st.write(f"--- {key.title()} {idx + 1} ---")

            # Create columns for actions
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                edited_item = {}
                for field_key, field_value in item.items():
                    edited_item[field_key] = DemoConfigEditor.render_field(field_key, field_value, f"{key}_{idx}")
                edited_items.append(edited_item)

            with col2:
                if st.button("🗑️ Delete", key=f"delete_{key}_{idx}"):
                    items.pop(idx)
                    st.rerun()

            with col3:
                if st.button("📋 Duplicate", key=f"dup_{key}_{idx}"):
                    items.insert(idx + 1, item.copy())
                    st.rerun()

        # Add new item
        st.write("---")
        if st.button(f"➕ Add New {key.title()}"):
            # Create template based on first item or empty dict
            template = items[0].copy() if items else {}
            for k in template:
                template[k] = "" if isinstance(template[k], str) else template[k]
            items.append(template)
            st.rerun()

        return items

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

        # Create tabs
        edit_tab, raw_tab, preview_tab = st.tabs(["📝 Edit", "📄 Raw YAML", "👁️ Preview"])

        with edit_tab:
            st.header(f"Editing: {selected_file.name}")

            edited_data = current_data.copy()

            # Iterate through top-level keys
            for key, value in current_data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    # Special handling for lists of dictionaries
                    edited_data[key] = DemoConfigEditor.render_list_editor(key, value)
                elif isinstance(value, dict):
                    # Handle dictionaries
                    st.subheader(key)
                    edited_data[key] = DemoConfigEditor.render_dict(key, value)
                else:
                    # Handle simple values
                    edited_data[key] = DemoConfigEditor.render_field(key, value)

        with raw_tab:
            st.header("Raw YAML Content")
            st.code(
                yaml.dump(current_data, default_flow_style=False, sort_keys=False, allow_unicode=True), language="yaml"
            )

        with preview_tab:
            st.header("Preview Edited Configuration")
            st.json(edited_data)

        # Save button
        st.sidebar.markdown("---")
        if st.sidebar.button("💾 Save Changes", type="primary", use_container_width=True):
            if DemoConfigEditor.save_yaml_file(selected_file, edited_data):
                st.sidebar.success("✅ Configuration saved successfully!")
                st.balloons()
            else:
                st.sidebar.error("❌ Failed to save configuration")

        # Reload button
        if st.sidebar.button("🔄 Reload from File", use_container_width=True):
            st.rerun()

        # Backup info
        st.sidebar.markdown("---")
        st.sidebar.info(f"**File Path:**\n`{selected_file}`\n\n**File Size:** {selected_file.stat().st_size:,} bytes")


if __name__ == "__main__":
    DemoConfigEditor.main()
