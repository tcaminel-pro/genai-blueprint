"""
The usual "tell me a joke" LLM call.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from python.ai_core.chain_registry import (
    Example,
    RunnableItem,
    register_runnable,
)
from python.ai_core.llm import get_llm
from python.ai_core.prompts import def_prompt

simple_prompt = """Tell me a joke on {topic}"""
joke_chain = (
    {"topic": RunnablePassthrough()}
    | def_prompt(user=simple_prompt)
    | get_llm()
    | StrOutputParser()
)


class Joke(BaseModel):
    the_joke: str = Field(description="a good joke")
    explanation: str = Field(description="explain why it's funny")
    rate: float = Field(description="rate how the joke is funny between 0 and 5")


parser = PydanticOutputParser(pydantic_object=Joke)

prompt_with_format = """
tell me  a joke on {topic};     
--- 
    {format_instructions}
"""

structured_prompt = def_prompt(user=prompt_with_format).partial(
    format_instructions=parser.get_format_instructions(),
)

parser = PydanticOutputParser(pydantic_object=Joke)
structured_joke = structured_prompt | get_llm() | parser


# Register the chain
register_runnable(
    RunnableItem(
        tag="Agent",
        name="Joke",
        runnable=structured_joke,
        examples=[Example(query=["Beaver"])],
    )
)
