# Does not seems to work on WSL. TBC

from smolagents import CodeAgent, DuckDuckGoSearchTool, GradioUI, LiteLLMModel, VisitWebpageTool

from src.ai_core.llm import LlmFactory

MODEL_ID = None

model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
print(model_name)
llm = LiteLLMModel(model_id=model_name)


agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=llm)

GradioUI(agent).launch()
