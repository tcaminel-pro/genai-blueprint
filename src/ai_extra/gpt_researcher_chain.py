"""GPT Researcher integration utilities.

This module provides simplified functions to run GPT Researcher with configurable parameters.
Main function is: run_gpt_researcher
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from gpt_researcher import GPTResearcher
except ImportError as ex:
    raise ImportError(
        "gpt-researcher package is required. Install with: uv add gpt-researcher --group ai_extra"
    ) from ex

from loguru import logger
from pydantic import BaseModel, Field

from src.ai_core.llm import LlmFactory


class ResearchReport(BaseModel):
    """Container class for GPT Researcher results and metadata."""

    report: str
    context: str
    costs: float = 0.0
    images: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)


def create_gptr_config(llm_id: str | None = None, **extra_params) -> str:
    """Create a temporary configuration file for GPT Researcher.

    Args:
        llm_id: LLM identifier to use for all models
        **extra_params: Additional configuration parameters

    Returns:
        Path to the temporary configuration file
    """
    config_dict = {}

    if llm_id:
        litellm_name = LlmFactory(llm_id=llm_id).get_litellm_model_name(separator=":")
        logger.info(f"Using LiteLLM model name: {litellm_name}")
        config_dict.update(
            {
                "FAST_LLM": litellm_name,
                "SMART_LLM": litellm_name,
                "STRATEGIC_LLM": litellm_name,
            }
        )

    config_dict.update(extra_params)

    path = Path(tempfile.gettempdir()) / "gptr_conf.json"
    with open(path, "w") as json_file:
        json.dump(config_dict, json_file, indent=2)
    logger.info(f"Created GPT Researcher config at {path}: {config_dict}")
    return str(path)


async def run_gpt_researcher(
    query: str, llm_id: str | None = None, verbose: bool = True, websocket_logger: Any | None = None, **kwargs
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
    # Extract config parameters
    max_iterations = kwargs.pop("max_iterations", 3)
    max_search_results = kwargs.pop("max_search_results_per_query", 5)

    try:
        config_path = create_gptr_config(
            llm_id=llm_id,
            MAX_ITERATIONS=max_iterations,
            MAX_SEARCH_RESULTS_PER_QUERY=max_search_results,
        )

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
            #llm_id=None,  # Use default models instead of potentially problematic custom LLM
            llm_id="gpt_41mini_openrouter",
            max_iterations=1,
            max_search_results_per_query=3,
            report_source="web",
            tone="Objective",
        )
        return result

    result = asyncio.run(main())
    print(result.report)
