"""Configuration loader for ReAct agent demos.

This module provides functionality for loading and processing
demo configurations from react_agent.yaml for the mcp_agent command.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.utils.config_mngr import global_config

REACT_CONF_YAML_FILE = "config/demos/react_agent.yaml"


class ReactDemo(BaseModel):
    """Configuration class for ReAct Agent demonstrations.

    This class defines the structure for setting up different demo scenarios
    for ReAct agents (LangChain-based) with MCP servers and tools.
    """

    name: str
    tools: List[str] = []
    mcp_servers: List[str] = []
    examples: List[str] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_react_demo_config(config_name: str) -> Optional[Dict[str, Any]]:
    """Load configuration for a specific ReAct demo from react_agent.yaml.

    Args:
        config_name: Name of the configuration to load (e.g., 'Weather')

    Returns:
        Dictionary containing the demo configuration, or None if not found
    """
    try:
        demos_config = global_config().merge_with(REACT_CONF_YAML_FILE).get_list("react_agent_demos")
        for demo_config in demos_config:
            if demo_config.get("name", "").lower() == config_name.lower():
                return demo_config
        return None
    except Exception as ex:
        logger.error(f"Failed to load ReAct demo config '{config_name}': {ex}")
        return None


def load_all_react_demos_from_config() -> List[ReactDemo]:
    """Load and configure all ReAct demonstration scenarios from the application configuration.

    Returns:
        List of configured ReactDemo objects
    """
    try:
        demos_config = global_config().merge_with(REACT_CONF_YAML_FILE).get_list("react_agent_demos")
        result = []

        for demo_config in demos_config:
            name = demo_config.get("name", "")
            examples = demo_config.get("examples", [])
            mcp_servers = demo_config.get("mcp_servers", [])
            tools = demo_config.get("tools", [])

            demo = ReactDemo(
                name=name,
                tools=tools,
                mcp_servers=mcp_servers,
                examples=examples,
            )
            result.append(demo)

        return result
    except Exception as ex:
        logger.error(f"Failed to load ReAct demo configurations: {ex}")
        return []


def create_react_demo_from_config(config_name: str) -> Optional[ReactDemo]:
    """Create a single ReactDemo from configuration.

    Args:
        config_name: Name of the configuration to load

    Returns:
        ReactDemo instance or None if not found
    """
    demo_config = load_react_demo_config(config_name)
    if not demo_config:
        return None

    name = demo_config.get("name", "")
    examples = demo_config.get("examples", [])
    mcp_servers = demo_config.get("mcp_servers", [])
    tools = demo_config.get("tools", [])

    return ReactDemo(
        name=name,
        tools=tools,
        mcp_servers=mcp_servers,
        examples=examples,
    )
