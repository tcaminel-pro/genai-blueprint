"""Shared configuration loader for SmolAgent demos.

This module provides common functionality for loading and processing
demo configurations from codeact_agent.yaml, eliminating code duplication
between the CLI and webapp implementations.
"""

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool as LangChainBaseTool
from loguru import logger
from pydantic import BaseModel, ConfigDict
from smolagents import Tool as SmolAgentTool

from src.utils.config_mngr import global_config, import_from_qualified

CONF_YAML_FILE = "config/demos/codeact_agent.yaml"


class SmolagentsAgentConfig(BaseModel):
    """Configuration class for CodeAct Agent demonstrations.

    This class defines the structure for setting up different demo scenarios
    including available tools, MCP servers, and example prompts.
    """

    name: str
    tools: List[Any] = []
    mcp_servers: List[str] = []
    examples: List[str] = []
    authorized_imports: List[str] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)


def process_tools_from_config(tools_config: List[Dict[str, Any]] | None) -> List[Any]:
    """Process tools configuration and return list of tool instances.

    Args:
        tools_config: List of tool configuration dictionaries, or None

    Returns:
        List of tool instances
    """
    tools = []

    # Handle None case (when YAML has "tools:" with no value)
    if tools_config is None:
        return tools

    for tool_config in tools_config:
        if not isinstance(tool_config, dict):
            continue

        try:
            if "function" in tool_config:
                tools.extend(_process_function_tool(tool_config))
            elif "class" in tool_config:
                tool_instance = _process_class_tool(tool_config)
                if tool_instance:
                    tools.append(tool_instance)
            elif "factory" in tool_config:
                tools.extend(_process_factory_tool(tool_config))
        except Exception as ex:
            logger.warning(f"Failed to process tool config {tool_config}: {ex}")

    return tools


def _process_function_tool(tool_config: Dict[str, Any]) -> List[Any]:
    """Process a function-based tool configuration."""
    func_ref = tool_config.get("function")
    tools = []

    if isinstance(func_ref, str) and ":" in func_ref:
        try:
            tool_func = import_from_qualified(func_ref)
            tools.append(tool_func)
        except Exception as ex:
            logger.warning(f"Failed to load function {func_ref}: {ex}")
    else:
        logger.warning(f"Unknown function reference: {func_ref}")

    return tools


def _process_class_tool(tool_config: Dict[str, Any]) -> Optional[Any]:
    """Process a class-based tool configuration."""
    class_ref = tool_config.get("class")
    params = {k: v for k, v in tool_config.items() if k != "class"}

    if isinstance(class_ref, str) and ":" in class_ref:
        try:
            tool_class = import_from_qualified(class_ref)
            return tool_class(**params)
        except Exception as ex:
            logger.warning(f"Failed to load class {class_ref}: {ex}")
    else:
        logger.warning(f"Unknown tool class reference: {class_ref}")

    return None


def _process_factory_tool(tool_config: Dict[str, Any]) -> List[Any]:
    """Process a factory-based tool configuration."""
    factory_ref = tool_config.get("factory")
    params = {k: v for k, v in tool_config.items() if k != "factory"}
    tools = []

    if isinstance(factory_ref, str) and ":" in factory_ref:
        try:
            factory_func = import_from_qualified(factory_ref)
            tool_result = factory_func(**params)

            if isinstance(tool_result, list):
                tools.extend(tool_result)
            else:
                tools.append(tool_result)
        except Exception as ex:
            logger.warning(f"Failed to load factory {factory_ref}: {ex}")
    else:
        logger.warning(f"Unknown factory reference: {factory_ref}")

    return tools


def convert_langchain_tools(tools: List[Any]) -> List[Any]:
    """Convert LangChain BaseTool instances to SmolAgent Tools.

    Args:
        tools: List of tool instances that may include LangChain tools

    Returns:
        List of tools with LangChain tools converted to SmolAgent tools
    """
    converted_tools = []
    for tool in tools:
        if isinstance(tool, LangChainBaseTool):
            converted_tools.append(SmolAgentTool.from_langchain(tool))
        else:
            converted_tools.append(tool)
    return converted_tools


def load_smolagent_demo_config(config_name: str) -> Optional[Dict[str, Any]]:
    """Load configuration for a specific demo from codeact_agent.yaml.

    Args:
        config_name: Name of the configuration to load (e.g., 'titanic')

    Returns:
        Dictionary containing the demo configuration, or None if not found
    """
    try:
        demos_config = global_config().merge_with(CONF_YAML_FILE).get_list("codeact_agent_demos")
        for demo_config in demos_config:
            if demo_config.get("name", "").lower() == config_name.lower():
                return demo_config
        return None
    except Exception as ex:
        logger.error(f"Failed to load demo config '{config_name}': {ex}")
        return None


def load_all_demos_from_config() -> List[SmolagentsAgentConfig]:
    """Load and configure all demonstration scenarios from the application configuration.

    Returns:
        List of configured CodeactDemo objects
    """
    try:
        demos_config = global_config().merge_with(CONF_YAML_FILE).get_list("codeact_agent_demos")
        result = []

        for demo_config in demos_config:
            name = demo_config.get("name", "")
            examples = demo_config.get("examples", [])
            mcp_servers = demo_config.get("mcp_servers", [])
            authorized_imports = demo_config.get("authorized_imports", [])

            # Process tools
            raw_tools = process_tools_from_config(demo_config.get("tools", []))
            converted_tools = convert_langchain_tools(raw_tools)

            demo = SmolagentsAgentConfig(
                name=name,
                tools=converted_tools,
                mcp_servers=mcp_servers,
                examples=examples,
                authorized_imports=authorized_imports,
            )
            result.append(demo)

        return result
    except Exception as ex:
        logger.exception(f"Failed to load demo configurations: {ex}")
        return []


def create_demo_from_config(config_name: str) -> Optional[SmolagentsAgentConfig]:
    """Create a single CodeactDemo from configuration.

    Args:
        config_name: Name of the configuration to load

    Returns:
        CodeactDemo instance or None if not found
    """
    demo_config = load_smolagent_demo_config(config_name)
    if not demo_config:
        return None

    name = demo_config.get("name", "")
    examples = demo_config.get("examples", [])
    mcp_servers = demo_config.get("mcp_servers", [])
    authorized_imports = demo_config.get("authorized_imports", [])

    # Process tools
    raw_tools = process_tools_from_config(demo_config.get("tools", []))
    converted_tools = convert_langchain_tools(raw_tools)

    return SmolagentsAgentConfig(
        name=name,
        tools=converted_tools,
        mcp_servers=mcp_servers,
        examples=examples,
        authorized_imports=authorized_imports,
    )
