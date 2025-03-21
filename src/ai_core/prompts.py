"""Prompt utilities and wrapper functions.

This module provides utilities for creating and managing LangChain prompts,
with support for system messages, user inputs, and additional message types.

Key Features:
- Automatic whitespace normalization
- System message support
- Flexible message composition
- Integration with LangChain's prompt templates

Example:
    >>> # Create simple prompt
    >>> prompt = def_prompt(
    ...     system="You are a helpful assistant",
    ...     user="Tell me a joke"
    ... )

    >>> # Create prompt with additional messages
    >>> prompt = def_prompt(
    ...     system="You are a math tutor",
    ...     user="Solve this problem: {problem}",
    ...     other_msg={"placeholder": "{scratchpad}"}
    ... )
"""

from textwrap import dedent

from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
)

DEFAULT_SYSTEM_PROMPT = ""


def dedent_ws(text: str) -> str:
    r"""'detent' function replacement to remove any common leading whitespace from every line in `text`.

    It address 'dedent' choice to not consider tabs and space as equivalent, by replacing tabs by 4 whitespace,
    so "   hello" and "\\thello" are considered to have common leading whitespace.

    It also remove the first new_line if any
    """
    # text = text.strip("\n")
    text = text.replace("\t", "    ")
    result = dedent(text).strip()
    return result


def def_prompt(system: str | None = None, user: str = "", other_msg: dict | None = None) -> BasePromptTemplate:
    """Small wrapper around 'ChatPromptTemplate.from_messages" with just a user  and optional system prompt and other messages.
    Common leading whitespace and tags are removed from the system and user strings.

    Example:
    .. code-block:: python
        prompt = def_prompt(system="You are an helpful agent", user = "bla bla", other_msg={"placeholder": "{agent_scratchpad}"})

    """
    if other_msg is None:
        other_msg = {}
    messages: list = []
    if system:
        messages.append(("system", dedent_ws(system)))
    messages.append(("user", dedent_ws(user)))
    other = list(other_msg.items())
    messages.extend(other)
    return ChatPromptTemplate.from_messages(messages)


def dict_input_message(user: str, system: str | None = None) -> dict[str, list[tuple]]:
    """
    Return an input message as dict, in the form : {"messages": [("user", query)]},  typically for use as input of a CompiledGraph.
    """
    msg = [("user", dedent_ws(user))]
    if system:
        msg += [("system", dedent_ws(system))]
    return {"messages": msg}


def list_input_message(user: str, system: str | None = None) -> list[dict[str, str]]:
    """
    Return an input message in the form: [{"role": "user", "content": query}]
    """
    msg = [{"role": "user", "content": dedent_ws(user)}]
    if system:
        msg += [{"role": "system", "content": dedent_ws(system)}]
    return msg
