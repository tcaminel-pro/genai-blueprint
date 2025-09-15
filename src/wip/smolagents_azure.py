import os

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    WebSearchTool,
)

from src.ai_core.llm_factory import LlmFactory

# Set environment variables that LiteLLM expects for Azure
if os.environ.get("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
if os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

llm_factory = LlmFactory(llm_tag="azure", llm_params={"temperature": 0.7})
llm = llm_factory.get_smolagent_model()
print(f"Using model: {llm_factory.get_id()}")
agent = CodeAgent(tools=[WebSearchTool()], model=llm)
agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
