"""Structured output utilities for LLM responses.

This module provides functions to handle structured output generation from language models,
supporting multiple methods like output parsing and structured output generation.

The main function `structured_output_chain` creates a chain that can generate structured
output according to a specified Pydantic model using different approaches

"""

from typing import Literal, Type, TypeVar

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from python.ai_core.llm import get_llm
from python.ai_core.prompts import def_prompt

T = TypeVar("T", bound=BaseModel)


METHODS = Literal["output_parser", "function_calling", "json_schema"]


def structured_output_chain(system: str, user: str, llm_id: str | None, output_class: type[T], method: METHODS):
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
    if method == "output_parser":
        parser = PydanticOutputParser(pydantic_object=output_class)
        chain = (
            def_prompt(system, user + "\n{format_instructions}").partial(
                format_instructions=parser.get_format_instructions()
            )
            | get_llm(llm_id=llm_id, json_mode=True)
            | parser
        )
    elif method in ["function_calling", "json_schema"]:
        llm = get_llm(llm_id=llm_id).with_structured_output(output_class, method=method)
        chain = def_prompt(system=system, user=user) | llm
    else:
        raise ValueError(f"Incorrect structured output method : {method}")
    return chain


if __name__ == "__main__":
    """Example usage demonstrating structured output generation."""

    class Joke(BaseModel):
        """Example model for a joke with setup and punchline."""

        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")

    MODEL = "gpt_4o_openai"
    MODEL = "claude_haiku35_openrouter"
    # MODEL = "deepseek_chatv3_deepseek"

    # a = structured_output_chain(
    #     system="",
    #     user="Tell me a joke about cats",
    #     llm_id=MODEL,
    #     output_class=Joke,
    #     method="output_parser",
    # )
    # debug(a.invoke({}))

    b = structured_output_chain(
        system="",
        user="Tell me a joke about cats",
        llm_id=MODEL,
        output_class=Joke,
        method="function_calling",
    )
    debug(b.invoke({}))
