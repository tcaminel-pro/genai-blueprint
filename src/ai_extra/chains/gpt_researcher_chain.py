"""GPT Researcher integration utilities.

This module provides simplified functions to run GPT Researcher with configurable parameters.
Main function is: run_gpt_researcher
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from src.ai_core.llm_factory import LlmFactory
from src.utils.config_mngr import global_config

try:
    from gpt_researcher import GPTResearcher
except ImportError as ex:
    raise ImportError(
        "gpt-researcher package is required. Install with: uv add gpt-researcher --group ai_extra"
    ) from ex

from loguru import logger
from pydantic import BaseModel, Field


class ResearchReport(BaseModel):
    """Container class for GPT Researcher results and metadata."""

    report: str
    context: str
    costs: float = 0.0
    images: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)


def create_gptr_config(config_name: str) -> str:
    """Get GPT Researcher config from our global configuration.

    Args:
        config_name: Name of the config section to use

    Returns:
        Path to the temporary configuration file with selected config
    """
    config_dict = global_config().get_dict(
        f"gpt_researcher.{config_name}",
        # expected_keys=["fast_llm"],
    )
    for llm in ["smart_llm", "fast_llm", "strategic_llm"]:
        if llm_id := config_dict.get(llm):
            litellm_name = LlmFactory(llm_id=llm_id).get_litellm_model_name(separator=":")
            config_dict[llm] = litellm_name
            logger.info(f"Using LiteLLM model name for {llm}: {litellm_name}")

    path = Path(tempfile.gettempdir()) / "gptr_conf.json"
    with open(path, "w") as json_file:
        json.dump(config_dict, json_file, indent=2)
    logger.info(f"Using GPT Researcher config '{config_name}': {config_dict}")
    return str(path)


async def run_gpt_researcher(
    query: str, config_name: str = "default", verbose: bool = True, websocket_logger: Any | None = None, **kwargs
) -> ResearchReport:
    """Execute a GPT Researcher task with configurable parameters.

    Args:
        query: Research query
        llm_id: LLM identifier to use
        verbose: Enable verbose output
        websocket_logger: Optional websocket logger
        **kwargs: Additional parameters for GPT Researcher

    Returns:
        ResearchReport: Container with research results and metadata
    """

    try:
        config_path = create_gptr_config(config_name)

        researcher = GPTResearcher(
            query=query, verbose=verbose, websocket=websocket_logger, config_path=config_path, **kwargs
        )

        logger.info(f"Starting GPT Researcher with query: {query}")
        await researcher.conduct_research()
        report = await researcher.write_report()

        return ResearchReport(
            report=report,
            context=str(researcher.get_research_context()),
            costs=researcher.get_costs(),
            images=[str(e) for e in researcher.get_research_images()],
            sources=researcher.get_research_sources(),
        )
    except Exception as e:
        logger.error(f"GPT Researcher failed: {e}")
        # Return a basic error report instead of crashing
        return ResearchReport(
            report=f"Research failed due to error: {str(e)}",
            context="Error occurred during research",
            costs=0.0,
            images=[],
            sources=[],
        )


# FOR QUICK TEST
if __name__ == "__main__":

    async def main():
        query = "what are the ethical risks of LLM powered AI Agents"
        result = await run_gpt_researcher(
            query=query,
        )
        return result

    result = asyncio.run(main())
    print(result.report)
