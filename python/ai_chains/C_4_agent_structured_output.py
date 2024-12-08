# https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/#define-graph

from typing import Literal

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from python.ai_core.custom_agents import create_react_structured_output_graph
from python.ai_core.llm import get_llm, llm_config


class WeatherResponse(BaseModel):
    """Respond to the user with this"""

    temperature: float = Field(description="The temperature in fahrenheit")
    wind_direction: str = Field(description="The direction of the wind in abbreviated form")
    wind_speed: float = Field(description="The speed of the wind in km/h")


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It is cloudy in NYC, with 10 mph winds in the North-East direction and a temperature of 66 degrees"
    elif city == "sf":
        return "It is 75 degrees and sunny in SF, with 3 mph winds in the South-East direction"
    else:
        raise AssertionError("Unknown city")


# @chain
def get_weather_fn(query: str, config: RunnableConfig) -> WeatherResponse:
    llm_id = config["configurable"].get("llm_id")
    llm = get_llm(llm_id)
    graph = create_react_structured_output_graph(llm, [get_weather], WeatherResponse)
    graph_input = {"messages": [("human", query)]}
    result = graph.invoke(graph_input)
    return result["final_response"]


MODEL = "claude_haiku35_openrouter"  # Work
MODEL = "nvidia_nemotrom70_openrouter"  # Do NOT work
MODEL = "gpt_4omini_openai"  # Work
MODEL = "google_gemini15flash_openrouter"  # Do NOT work
MODEL = "qwen25_72_openrouter"  # Do NOT work
MODEL = "gpt_4omini_edenai"  # ?
MODEL = "llama32_3_ollama"  # Do NOT work
MODEL = "mistral_nemo_openrouter"  # Work
MODEL = "google_gemini15pro_openrouter"  # Work
MODEL = "llama33_70_deepinfra"  # Does NOT work
MODEL = "llama33_70_groq"  # Work !!
MODEL = "gpt_4o_azure"  # Work, but need to remove tool_choice="any" with old API
MODEL = "llama31_405_deepinfra" # Does NOT work


get_weather_chain = RunnableLambda(get_weather_fn).with_config(llm_config(MODEL) | {"recursion_limit": 6})
answer = get_weather_chain.invoke("what's the weather in New York?")

debug(answer)
