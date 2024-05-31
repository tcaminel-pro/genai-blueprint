# might be incomplete
# see https://python.langchain.com/v0.1/docs/modules/agents/agent_types/


from typing import Callable, Sequence

from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_openai_tools_agent,
    create_structured_chat_agent,
    create_tool_calling_agent,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic.v1 import BaseModel


class AgentBuilder(BaseModel):
    type: str
    create_function: Callable[
        [BaseLanguageModel, Sequence[BaseTool], ChatPromptTemplate], Runnable
    ]
    hub_prompt: str

    def get_agent_executor(
        self, llm: BaseLanguageModel, tools=Sequence[BaseTool]
    ) -> AgentExecutor:
        prompt = hub.pull(self.hub_prompt)
        agent = self.create_function(llm, tools, prompt)  # type: ignore
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # type: ignore
        return agent_executor


AGENT_BUILDER_LIST = [
    AgentBuilder(
        type="tool_calling",
        create_function=create_tool_calling_agent,
        hub_prompt="hwchase17/openai-tools-agent",
    ),
    AgentBuilder(
        type="openai_tool",
        create_function=create_openai_tools_agent,
        hub_prompt="hwchase17/openai-tools-agent",
    ),
    AgentBuilder(
        type="openai_function",
        create_function=create_openai_functions_agent,
        hub_prompt="hwchase17/openai-functions-agent",
    ),
    AgentBuilder(
        type="structured_chat",
        create_function=create_structured_chat_agent,
        hub_prompt="hwchase17/structured-chat-agent",
    ),
]


def get_agent_builder(type: str) -> AgentBuilder:
    agent = next((x for x in AGENT_BUILDER_LIST if x.type == type), None)
    if agent is None:
        raise ValueError(f"Unknown agent type: {type}")
    return agent
