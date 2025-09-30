"""Shared configuration loader for LangChain-based agents.

This module provides common functionality for loading and processing
configurations for LangChain-based agents like react_agent and list_deep_agents,
reducing code duplication and providing a consistent interface.
"""

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.utils.config_mngr import global_config, import_from_qualified


class LangChainAgentConfig(BaseModel):
    """Configuration class for LangChain-based agent demonstrations.

    This class defines the structure for setting up different demo scenarios
    including available tools, MCP servers, and example prompts.
    """

    name: str
    tools: List[BaseTool] = []
    tool_configs: List[Dict[str, Any]] = []  # Raw tool configurations
    mcp_servers: List[str] = []
    examples: List[str] = []
    system_prompt: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


def process_langchain_tools_from_config(
    tools_config: List[Dict[str, Any]] | None, llm: Optional[str] = None
) -> List[BaseTool]:
    """Process tools configuration and return list of LangChain tool instances.

    Args:
        tools_config: List of tool configuration dictionaries, or None
        llm: Optional LLM instance to pass to factory functions that support it

    Returns:
        List of LangChain BaseTool instances
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
                tools.extend(_process_factory_tool(tool_config, llm=llm))
        except Exception as ex:
            raise Exception(f"Failed to process tool config {tool_config}: {ex}") from ex
            logger.warning(f"Failed to process tool config {tool_config}: {ex}")

    return tools


def _process_function_tool(tool_config: Dict[str, Any]) -> List[BaseTool]:
    """Process a function-based tool configuration."""
    func_ref = tool_config.get("function")
    tools = []

    if isinstance(func_ref, str) and ":" in func_ref:
        try:
            tool_func = import_from_qualified(func_ref)
            # If it's a function that returns a tool, call it
            if callable(tool_func):
                # Check if it's already a tool instance
                if isinstance(tool_func, BaseTool):
                    tools.append(tool_func)
                else:
                    # It's a function that might return a tool
                    result = tool_func()
                    if isinstance(result, BaseTool):
                        tools.append(result)
                    elif isinstance(result, list):
                        tools.extend([t for t in result if isinstance(t, BaseTool)])
            else:
                # It's already a tool instance
                if isinstance(tool_func, BaseTool):
                    tools.append(tool_func)
        except Exception as ex:
            raise Exception(f"Failed to load  {func_ref}: {ex}") from ex
            logger.warning(f"Failed to load function {func_ref}: {ex}")
    else:
        logger.warning(f"Unknown function reference: {func_ref}")

    return tools


def _process_class_tool(tool_config: Dict[str, Any]) -> Optional[BaseTool]:
    """Process a class-based tool configuration."""
    class_ref = tool_config.get("class")
    params = {k: v for k, v in tool_config.items() if k != "class"}

    if isinstance(class_ref, str) and ":" in class_ref:
        try:
            tool_class = import_from_qualified(class_ref)
            instance = tool_class(**params)
            if isinstance(instance, BaseTool):
                return instance
            else:
                logger.warning(f"Class {class_ref} does not produce a BaseTool instance")
        except Exception as ex:
            logger.warning(f"Failed to load class {class_ref}: {ex}")
    else:
        logger.warning(f"Unknown tool class reference: {class_ref}")

    return None


def _process_factory_tool(tool_config: Dict[str, Any], llm: Optional[str] = None) -> List[BaseTool]:
    """Process a factory-based tool configuration.

    Args:
        tool_config: Tool configuration dictionary
        llm: Optional LLM instance to pass to factory functions that support it
    """
    factory_ref = tool_config.get("factory")
    params = {k: v for k, v in tool_config.items() if k != "factory"}
    tools = []

    if isinstance(factory_ref, str) and ":" in factory_ref:
        try:
            factory_func = import_from_qualified(factory_ref)

            # If LLM is provided and factory function accepts it, pass it along
            import inspect

            sig = inspect.signature(factory_func)
            if llm is not None and "llm" in sig.parameters:
                params["llm"] = llm

            tool_result = factory_func(**params)

            if isinstance(tool_result, list):
                tools.extend([t for t in tool_result if isinstance(t, BaseTool)])
            elif isinstance(tool_result, BaseTool):
                tools.append(tool_result)
        except Exception as ex:
            logger.warning(f"Failed to load factory {factory_ref}: {ex}")
    else:
        logger.warning(f"Unknown factory reference: {factory_ref}")

    return tools


def load_langchain_agent_config(config_file: str, config_section: str, config_name: str) -> Optional[Dict[str, Any]]:
    """Load configuration for a specific agent from a YAML file.

    Args:
        config_file: Path to the YAML configuration file
        config_section: Section name in the YAML file (e.g., 'react_agent_demos')
        config_name: Name of the configuration to load (e.g., 'Weather')

    Returns:
        Dictionary containing the configuration, or None if not found
    """
    try:
        demos_config = global_config().merge_with(config_file).get_list(config_section)
        for demo_config in demos_config:
            if demo_config.get("name", "").lower() == config_name.lower():
                return demo_config
        return None
    except Exception as ex:
        logger.error(f"Failed to load agent config '{config_name}' from {config_file}: {ex}")
        return None


def create_langchain_agent_config(
    config_file: str, config_section: str, config_name: str, llm: Optional[Any] = None
) -> Optional[LangChainAgentConfig]:
    """Create a LangChainAgentConfig from YAML configuration.

    Args:
        config_file: Path to the YAML configuration file
        config_section: Section name in the YAML file
        config_name: Name of the configuration to load
        llm: Optional LLM instance to pass to factory functions that support it

    Returns:
        LangChainAgentConfig instance or None if not found
    """
    demo_config = load_langchain_agent_config(config_file, config_section, config_name)
    if not demo_config:
        return None

    name = demo_config.get("name", "")
    examples = demo_config.get("examples", [])
    mcp_servers = demo_config.get("mcp_servers", [])
    tool_configs = demo_config.get("tools", [])
    system_prompt = demo_config.get("system_prompt")

    # Process tools with the provided LLM
    processed_tools = process_langchain_tools_from_config(tool_configs, llm=llm)

    return LangChainAgentConfig(
        name=name,
        tools=processed_tools,
        tool_configs=tool_configs,
        mcp_servers=mcp_servers,
        examples=examples,
        system_prompt=system_prompt,
    )


def load_all_langchain_agent_configs(config_file: str, config_section: str) -> List[LangChainAgentConfig]:
    """Load and configure all demonstration scenarios from a YAML file.

    Args:
        config_file: Path to the YAML configuration file
        config_section: Section name in the YAML file

    Returns:
        List of configured LangChainAgentConfig objects
    """
    try:
        demos_config = global_config().merge_with(config_file).get_list(config_section)
        result = []

        for demo_config in demos_config:
            name = demo_config.get("name", "")
            examples = demo_config.get("examples", [])
            mcp_servers = demo_config.get("mcp_servers", [])
            tool_configs = demo_config.get("tools", [])
            system_prompt = demo_config.get("system_prompt")

            # Process tools
            processed_tools = process_langchain_tools_from_config(tool_configs)

            config = LangChainAgentConfig(
                name=name,
                tools=processed_tools,
                tool_configs=tool_configs,
                mcp_servers=mcp_servers,
                examples=examples,
                system_prompt=system_prompt,
            )
            result.append(config)

        return result
    except Exception as ex:
        logger.exception(f"Failed to load agent configurations from {config_file}: {ex}")
        return []
