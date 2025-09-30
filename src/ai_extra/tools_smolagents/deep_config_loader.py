"""Configuration loader for Deep Agent demos.

This module provides functionality for loading and processing
demo configurations from deep_agent.yaml for the deep-agent command.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.utils.config_mngr import global_config

DEEP_AGENT_CONF_YAML_FILE = "config/demos/deep_agent.yaml"


class DeepAgentDemo(BaseModel):
    """Configuration class for Deep Agent demonstrations.

    This class defines the structure for setting up different demo scenarios
    for Deep agents with custom instructions, tools, and settings.
    """

    name: str
    instructions: str
    tools: List[Any] = []
    mcp_servers: List[str] = []
    enable_file_system: bool = True
    enable_planning: bool = True
    examples: List[str] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_deep_agent_demo_config(config_name: str) -> Optional[Dict[str, Any]]:
    """Load configuration for a specific Deep Agent demo from deep_agent.yaml.

    Args:
        config_name: Name of the configuration to load (e.g., 'Research')

    Returns:
        Dictionary containing the demo configuration, or None if not found
    """
    try:
        demos_config = global_config().merge_with(DEEP_AGENT_CONF_YAML_FILE).get_list("deep_agent_demos")
        for demo_config in demos_config:
            if demo_config.get("name", "").lower() == config_name.lower():
                return demo_config
        return None
    except Exception as ex:
        logger.error(f"Failed to load Deep Agent demo config '{config_name}': {ex}")
        return None


def load_all_deep_agent_demos_from_config() -> List[DeepAgentDemo]:
    """Load and configure all Deep Agent demonstration scenarios from the application configuration.

    Returns:
        List of configured DeepAgentDemo objects
    """
    try:
        demos_config = global_config().merge_with(DEEP_AGENT_CONF_YAML_FILE).get_list("deep_agent_demos")
        result = []

        for demo_config in demos_config:
            name = demo_config.get("name", "")
            instructions = demo_config.get("instructions", "")
            tools = demo_config.get("tools", [])
            mcp_servers = demo_config.get("mcp_servers", [])
            enable_file_system = demo_config.get("enable_file_system", True)
            enable_planning = demo_config.get("enable_planning", True)
            examples = demo_config.get("examples", [])

            demo = DeepAgentDemo(
                name=name,
                instructions=instructions,
                tools=tools,
                mcp_servers=mcp_servers,
                enable_file_system=enable_file_system,
                enable_planning=enable_planning,
                examples=examples,
            )
            result.append(demo)

        return result
    except Exception as ex:
        logger.error(f"Failed to load Deep Agent demo configurations: {ex}")
        return []


def create_deep_agent_demo_from_config(config_name: str) -> Optional[DeepAgentDemo]:
    """Create a single DeepAgentDemo from configuration.

    Args:
        config_name: Name of the configuration to load

    Returns:
        DeepAgentDemo instance or None if not found
    """
    demo_config = load_deep_agent_demo_config(config_name)
    if not demo_config:
        return None

    name = demo_config.get("name", "")
    instructions = demo_config.get("instructions", "")
    tools = demo_config.get("tools", [])
    mcp_servers = demo_config.get("mcp_servers", [])
    enable_file_system = demo_config.get("enable_file_system", True)
    enable_planning = demo_config.get("enable_planning", True)
    examples = demo_config.get("examples", [])

    return DeepAgentDemo(
        name=name,
        instructions=instructions,
        tools=tools,
        mcp_servers=mcp_servers,
        enable_file_system=enable_file_system,
        enable_planning=enable_planning,
        examples=examples,
    )
