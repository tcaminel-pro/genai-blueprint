"""LangChain Runnable registry and management system.

This module implements a centralized registry for managing LangChain Runnables,
providing a unified interface for registration, retrieval, and execution of
various chain components.

Key Features:
- Centralized registry for all Runnable components
- Support for multiple Runnable types (instances, factories, key-based pairs)
- Example-based testing and demonstration
- Dynamic module loading for chain definitions
- Metadata and diagram support for documentation

The registry supports three types of Runnables:
1. Direct Runnable instances
2. Factory functions returning Runnables
3. Key-based callable pairs for parameterized Runnables

Example:
    >>> # Register a new chain
    >>> register_runnable(RunnableItem(
    ...     name="my_chain",
    ...     runnable=my_chain_instance,
    ...     examples=[Example(query=["sample query"])]
    ... ))

    >>> # Find and execute a chain
    >>> chain = find_runnable("my_chain")
    >>> result = chain.invoke("input text")
"""

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

from langchain_core.runnables import Runnable, RunnableLambda
from loguru import logger
from pydantic import BaseModel, ConfigDict, FilePath

from python.config import Config


class Example(BaseModel):
    """Defines examples for demonstrating and testing Runnable behavior.

    This class encapsulates example queries and optional file paths for
    demonstration purposes, particularly useful for RAG (Retrieval Augmented Generation)
    scenarios.

    Attributes:
        query (list[str]): List of example queries or prompts
        path (FilePath | None): Optional path to a file containing additional examples or context
    """

    query: list[str]
    path: FilePath | None = None
    # ext: str | None = None


class RunnableItem(BaseModel):
    """A comprehensive wrapper for LangChain Runnable components with metadata and execution capabilities.

    This class encapsulates a LangChain Runnable along with associated metadata and provides
    methods for executing the Runnable with various configurations. It supports multiple
    forms of Runnable definitions and includes demo/testing capabilities.

    Attributes:
        name (str): Unique identifier for the Runnable
        tag (str | None): Optional categorization tag
        runnable (Union[Runnable, Tuple[str, Callable], Callable]): The actual Runnable component,
            which can be:
            - A direct Runnable instance
            - A factory function returning a Runnable
            - A tuple of (key, factory_function) where the function creates a Runnable
              expecting a dict with the specified key
        examples (list[Example]): List of example inputs for testing/demo
        diagram (str | None): Optional diagram showing the Runnable's structure/flow

    Methods:
        invoke: Execute the Runnable with a single input
        stream: Stream the Runnable's execution results
        get: Retrieve the configured Runnable instance
    """

    name: str
    tag: str | None = None
    runnable: (
        Runnable | tuple[str, Callable[[dict[str, Any]], Runnable]] | Callable[[dict[str, Any]], Runnable]
    )  # Either a Runnable, or ...
    examples: list[Example] = []
    diagram: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def invoke(self, input: str, conf: dict[str, Any]) -> Any:
        runnable = self.get(conf)
        # is_agent = isinstance(runnable, AgentExecutor)
        runnable = runnable.with_config(configurable=conf)
        result = runnable.invoke(input)
        return result

    def stream(self, input: str, conf: dict[str, Any]) -> Iterator:
        runnable = self.get(conf)
        # is_agent = isinstance(runnable, AgentExecutor)
        runnable = runnable.with_config(configurable=conf)
        result = runnable.stream(input)
        return result

    def get(self, conf={"llm": None}) -> Runnable:
        if isinstance(self.runnable, Runnable):
            runnable = self.runnable
        elif isinstance(self.runnable, Callable):
            runnable = self.runnable(conf)
        elif isinstance(self.runnable, tuple):
            key, func = self.runnable
            func_runnable = _to_key_param_callable(key, func)
            runnable = func_runnable(conf)
        else:
            raise Exception("unknown or ill-formatted Runnable")
        # debug(self.runnable, runnable)
        return runnable


# Global registry
_registry: list[RunnableItem] = []


def register_runnable(r: RunnableItem) -> None:
    """Register a new RunnableItem in the global registry.

    Args:
        r (RunnableItem): The Runnable item to register
    """
    _registry.append(r)


def get_runnable_registry() -> list[RunnableItem]:
    """Retrieve the complete list of registered Runnables.

    Returns:
        list[RunnableItem]: List of all registered Runnable items
    """
    return _registry


def find_runnable(name: str) -> RunnableItem | None:
    """Find a registered Runnable by its name (case-insensitive).

    Args:
        name (str): Name of the Runnable to find

    Returns:
        RunnableItem | None: The matching Runnable item or None if not found
    """
    return next((x for x in _registry if x.name.strip().lower() == name.strip().lower()), None)


def _to_key_param_callable(key: str, function: Callable[[dict[str, Any]], Runnable]) -> Callable[[Any], Runnable]:
    """Convert a key-based function to a callable that works with the Runnable pipeline.

    This helper function transforms a function that expects a configuration dictionary
    and returns a Runnable into a pipeline-compatible function that automatically
    wraps string inputs into the expected dictionary format.

    Args:
        key (str): The dictionary key to use for the input value
        function (Callable): The original function that creates a Runnable

    Returns:
        Callable[[Any], Runnable]: A wrapped function that creates a properly configured Runnable
    """
    return lambda conf: RunnableLambda(lambda x: {key: x}) | function(conf)


def load_modules_with_chains() -> None:
    """Dynamically load chain modules specified in the configuration.

    This function reads the configuration to find and import modules containing
    chain definitions. It uses the 'chains.path' and 'chains.modules' configuration
    values to determine which modules to load.

    Raises:
        AssertionError: If the specified path doesn't exist
        Exception: If module loading fails (logged as warning)
    """
    config = Config.singleton()
    path = config.get_str("chains", "path")
    modules = config.get_list("chains", "modules")
    assert Path(path).exists

    for module in modules:
        try:
            importlib.import_module(f"{path}.{module}")
        except Exception as ex:
            logger.warning(f"Cannot load module {module}: {ex}")
