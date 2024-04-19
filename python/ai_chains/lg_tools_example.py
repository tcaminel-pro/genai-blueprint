"""

Adapted from https://blog.langchain.dev/tool-calling-with-langchain/  
"""

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from python.config import get_config
from python.ai_core.embeddings import get_embeddings
from python.ai_core.llm import llm_factory, set_cache
from python.ai_core.vector_store import get_vector_store


from python.ai_core.chain_registry import register_runnable
from python.config import get_config
from python.ai_core.embeddings import get_embeddings
from python.ai_core.llm import llm_getter
from python.ai_core.vector_store import get_vector_store


# ✅ Pydantic class
class multiply(BaseModel):
    """Return product of 'x' and 'y'."""

    x: float = Field(..., description="First factor")
    y: float = Field(..., description="Second factor")


# ✅ LangChain tool
@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y


# ✅ Function


def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x


# ✅ OpenAI-format dict
# Could also pass in a JSON schema with "title" and "description"
add = {
    "name": "add",
    "description": "Add 'x' and 'y'.",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "First number to add"},
            "y": {"type": "number", "description": "Second number to add"},
        },
        "required": ["x", "y"],
    },
}


# Whenever we invoke `llm_with_tool`, all three of these tool definitions
# are passed to the model.
llm_with_tools = llm_getter().bind(stop="STOP")


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_with_tools = llm.bind_tools([multiply, exponentiate, add, subtract])


register_runnable(
    "Tool",
    "Simple calculator",
    llm_with_tools,
    examples=["what's 5 raised to the 2.743"],
)

