"""
The usual "tell me a joke" LLM call.
"""

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from python.ai_core.chain_registry import (
    Example,
    RunnableItem,
    register_runnable,
)
from python.ai_core.llm import get_llm
from python.ai_core.prompts import def_prompt


class Joke(BaseModel):
    the_joke: str = Field(description="a good joke")
    explanation: str = Field(description="explain why it's funny")
    rate: float = Field(description="rate how the joke is funny between 0 and 5")


parser = PydanticOutputParser(pydantic_object=Joke)

user = """
tell me  a joke on {topic};     
--- 
    {format_instructions}
"""

prompt = def_prompt(user=user).partial(
    format_instructions=parser.get_format_instructions(),
)

parser = PydanticOutputParser(pydantic_object=Joke)
chain = prompt | get_llm() | parser

# Register the chain
register_runnable(
    RunnableItem(
        tag="Agent",
        name="Joke",
        runnable=chain,
        examples=[Example(query=["Beaver"])],
    )
)
