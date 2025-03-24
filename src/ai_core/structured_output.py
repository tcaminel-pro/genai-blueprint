"""Structured output utilities for LLM responses.

This module provides functions to handle structured output generation from language models,
supporting multiple methods like output parsing and structured output generation.

The main function `structured_output_chain` creates a chain that can generate structured
output according to a specified Pydantic model using different approaches

"""

from enum import Enum
from typing import TypeVar

from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.ai_core.llm import get_llm
from src.ai_core.prompts import def_prompt

T = TypeVar("T", bound=BaseModel)


class StructOutMethod(Enum):
    """Enumeration of supported structured output methods."""

    OUTPUT_PARSER = "output_parser"
    FUNCTION_CALLING = "function_calling"
    JSON_SCHEMA = "json_schema"
    AUTO = "auto"


def structured_output_chain(
    system: str,
    user: str,
    llm_id: str | None,
    output_class: type[T],
    method: StructOutMethod = StructOutMethod.FUNCTION_CALLING,
) -> Runnable[dict, T]:
    """Create a chain that generates structured output according to a Pydantic model.

    Methods can be:
    - output_parser: Uses LangChain's PydanticOutputParser
    - function_calling: Uses LLM's native function calling capabilities
    - json_schema: Uses LLM's JSON schema generation  (only supported by recent OpenAI models, as of 01/2023)

    Args:
        system: System message for the LLM
        user: User prompt template
        llm_id: Identifier for the LLM to use
        output_class: Pydantic model class for the output structure
        method: Which structured output method to use

    Returns:
        A configured LangChain chain ready for invocation


    Example:
        ```python
        chain = structured_output_chain(
            system="You are a helpful assistant",
            user="Tell me a joke about cats",
            llm_id="gpt-4",
            output_class=Joke,
            method="output_parser"
        )
        result = chain.invoke({})
        ```
    """
    if method == StructOutMethod.AUTO:
        raise NotImplementedError("Default selection of structured output not implemented yet")

    if method == StructOutMethod.OUTPUT_PARSER:
        parser = PydanticOutputParser(pydantic_object=output_class)
        chain = (
            def_prompt(system, user + "\n{format_instructions}").partial(
                format_instructions=parser.get_format_instructions()
            )
            | get_llm(llm_id=llm_id, json_mode=True)
            | parser
        )
    elif method in [StructOutMethod.FUNCTION_CALLING, StructOutMethod.JSON_SCHEMA]:
        llm = get_llm(llm_id=llm_id).with_structured_output(output_class, method=method.value)
        chain = def_prompt(system=system, user=user) | llm
    else:
        raise ValueError(f"Incorrect structured output method : {method}")
    return chain  # type: ignore


if __name__ == "__main__":
    """Example usage demonstrating structured output generation."""

    class Joke(BaseModel):
        """Example model for a joke with setup and punchline."""

        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")

    MODEL = "gpt_4o_openai"
    MODEL = "claude_haiku35_openrouter"
    MODEL = "deepseek_chatv3_deepseek"
    # MODEL = "llama33_70_deepinfra"  # function_calling NOT WORKING
    # MODEL = "nvidia_nemotrom70_openrouter"  # function_calling NOT WORKING
    # MODEL = "llama31_70_deepinfra"  # function_calling NOT WORKING
    a = structured_output_chain(
        system="",
        user="Tell me a joke about {topic}",
        llm_id=MODEL,
        output_class=Joke,
        method=StructOutMethod.OUTPUT_PARSER,
    )
    rprint(a.invoke({"topic": "cat"}))

    b = structured_output_chain(
        system="",
        user="Tell me a joke about {topic}",
        llm_id=MODEL,
        output_class=Joke,
        method=StructOutMethod.FUNCTION_CALLING,
    )
    rprint(b.invoke({"topic": "cat"}))
