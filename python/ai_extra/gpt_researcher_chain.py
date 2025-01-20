"""
GPT Researcher integration utilities.

This module provides classes and functions to manage and configure GPT Researcher's
language models, settings, and research operations. It handles:

- Configuration of LLM models and embeddings
- Creation of temporary configuration files
- Execution of research tasks with configurable parameters
- Management of research reports and metadata

Main function is: gpt_researcher_chain

"""

import asyncio
import json
import tempfile
import textwrap
from pathlib import Path
from typing import Any, List, Literal

try:
    from gpt_researcher import GPTResearcher
except ImportError:
    raise ImportError("gpt-researcher package is required. Install with: poetry add gpt-researcher --group ai_extra")
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from loguru import logger
from pydantic import BaseModel, Field

from python.ai_core.llm import LlmFactory
from python.utils.pydantic.kv_store import read_pydantic_from_store, save_pydantic_to_store


class GptrConfVariables(BaseModel):
    """Essentially a mapping in a class of GPT Researcher's configuration variables, normaly set done through
    environment variables or a JSON file . AllAlso used to define LLM by our llm-id. */.

    Attributes:
        fast_llm_id (str | None): Identifier for the fast language model.
        smart_llm_id (str | None): Identifier for the smart language model.
        strategic_llm_id (str | None): Identifier for the strategic language model.
        embeddings_id (str | None): Identifier for the embeddings model (not yet implemented).
        extra_params (dict): Additional configuration variables to include.

    Note:
        The LLM identifiers are in ours. They are converted to LiteLLM names.
    """

    fast_llm_id: str | None = None
    smart_llm_id: str | None = None
    strategic_llm_id: str | None = None
    embeddings_id: str | None = None
    extra_params: dict = {}

    def get_config_path(self) -> str:
        """Create a temporary configuration file with GPT Researcher configuration settings, that override the default one

        (See  https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/config/variables/default.py)
        The configuration file is created in the system's temporary directory.

        Example:
        >>> config = GptResearcherConf(
        ...     fast_llm_id = "gpt_4omini_openai",
        ...     smart_llm_id="gpt_4_openai,"
                extra_param = {
                    "MAX_ITERATIONS": 1,
                    "MAX_SEARCH_RESULTS_PER_QUERY": 3
                }
        ... )
        """

        config_dict = {}
        if self.embeddings_id is not None:
            raise NotImplementedError("configuring Embeddings model not yet implemented....")

        if self.fast_llm_id:
            config_dict["FAST_LLM"] = "litellm:" + LlmFactory(llm_id=self.fast_llm_id).get_litellm_model_name()

        if self.smart_llm_id:
            config_dict["SMART_LLM"] = "litellm:" + LlmFactory(llm_id=self.smart_llm_id).get_litellm_model_name()

        if self.strategic_llm_id:
            config_dict["STRATEGIC_LLM"] = (
                "litellm:" + LlmFactory(llm_id=self.strategic_llm_id).get_litellm_model_name()
            )

        config_dict |= self.extra_params

        path = Path(tempfile.gettempdir()) / "gptr_conf.json"
        with open(path, "w") as json_file:
            json.dump(config_dict, json_file)
        return str(path)


class CommonConfigParams(BaseModel):
    " NOT USED YET "
    # https://docs.gptr.dev/docs/gpt-researcher/gptr/config

    report_type: Literal["research_report", "detailed_report", "outline_report", "custom_report"] = "research_report"
    tone: Literal["Objective", "Analytical", "Informative", "Formal", "Explanatory", "Descriptive"] = "Objective"
    retriever: set[Literal["tavily", "duckduckgo", "google", "bing", "arxiv", "serpapi", "pubmed_central"]] = {"tavily"}
    sources: list[str] = []
    langage: Literal["english", "french", "dutch", "spanish", "german"] = "english"
    max_iteration: int = Field(gt=0, lt=5, default=4)
    max_search_result_per_query: int = Field(gt=0, lt=10, default=5)
    curate_sources: bool = True
    max_subtomic: int = Field(gt=0, lt=5, default=3)
    temprature: float = Field(ge=0.0, le=1.0, default=0.55)


class ResearchReport(BaseModel):
    """Container class for GPT Researcher results and metadata.

    Attributes:
        report (str): Detailed research report text
        context (str): Additional research context
        costs (float): Research-related cost information
        images (List[str]): Images related to the research
        sources (List[dict]): List of research sources and references
    """

    report: str
    context: str
    costs: float = 0.0
    images: List[str] = Field(default_factory=list)
    sources: List[dict] = Field(default_factory=list)


async def run_gpt_researcher(
    query: str, gptr_config: GptrConfVariables, verbose: bool = True, gptr_logger: Any | None = None, **kwargs
) -> ResearchReport:
    """Execute a GPT Researcher task with configurable parameters.

    Returns:
        ResearchReport: Container with research results and metadata
    """

    researcher = GPTResearcher(
        query=query, verbose=verbose, websocket=gptr_logger, config_path=gptr_config.get_config_path(), **kwargs
    )
    logger.info(f"start GPT Researcher {kwargs} {gptr_config}")
    _ = await researcher.conduct_research()
    report = await researcher.write_report()

    return ResearchReport(
        report=report,
        context=str(researcher.get_research_context()),
        costs=researcher.get_costs(),
        images=[str(e) for e in researcher.get_research_images()],
        sources=researcher.get_research_sources(),
    )


def gpt_researcher_chain() -> Runnable[str, ResearchReport]:
    """
    LCEL wrapper for the async research function, usable as Langchain callable.

    Returns:
        ResearchReport: Comprehensive research report.

    Configurable options:
        - logger: Optional logger instance for websocket communication
        - use_cached_result: Boolean to enable/disable result caching
        - gptr_conf: GptrConf instance for GPT Researcher configuration
        - gptr_params: Dictionary of additional parameters for research execution

    Example:
    ```python
        researcher_conf = GptrConf(
            fast_llm_id="gpt_4omini_openrouter",
            smart_llm_id="gpt_4_openrouter",
            extra_params={"MAX_ITERATIONS": 1, "MAX_SEARCH_RESULTS_PER_QUERY": 3}
        )
        chain = gpt_researcher_chain().with_config({
            "configurable": {
                "logger": None,
                "use_cached_result": True,
                "gptr_conf": researcher_conf,
                "gptr_params": {"report_source": "web", "tone": "Objective"}
            }
        })
        ```
    """

    async def fn(query: str | None, config: RunnableConfig) -> ResearchReport:
        gptr_logger = config["configurable"].get("logger")
        use_cached_result = config["configurable"].get("use_cached_result", False)
        gptr_conf = config["configurable"].get("gptr_conf", {})
        gptr_params = config["configurable"].get("gptr_params", {})
        result_key = config["configurable"].get("result_key", None)

        # Validate configurable keys
        allowed_keys = {"logger", "use_cached_result", "gptr_conf", "gptr_params", "result_key"}
        config_keys = set(config["configurable"].keys())
        if not config_keys.issubset(allowed_keys):
            raise ValueError(
                f"Invalid configurable keys: {config_keys - allowed_keys}. Allowed keys are: {allowed_keys}"
            )

        if query and gptr_params.get("query"):
            logger.warning("Query set twice for GPT Researcher")

        if query is None or query == "":
            if param_query := gptr_params.pop("query"):
                query = param_query
            else:
                raise ValueError("No query provided")
        assert query, "no query set for GPTR search"
        kv_store_key = result_key or query

        if use_cached_result:
            cached_result = read_pydantic_from_store(model_class=ResearchReport, key=kv_store_key)
            if cached_result:
                logger.info(f"use cached research report for query: '{textwrap.shorten(query, 15)}'")
                return cached_result

        result = await run_gpt_researcher(query=query, **gptr_params, gptr_config=gptr_conf, gptr_logger=gptr_logger)
        if use_cached_result:
            save_pydantic_to_store(kv_store_key, result)
        return result

    return RunnableLambda(func=fn)


# FOR QUICK TEST
if __name__ == "__main__":

    async def main():
        query = "what are the ethical risks of LLM powered AI Agents"
        gpt_llm = "gpt_4omini_openrouter"
        researcher_conf = GptrConfVariables(
            fast_llm_id=gpt_llm,
            smart_llm_id=gpt_llm,
            strategic_llm_id=gpt_llm,
            extra_params={"MAX_ITERATIONS": 1, "MAX_SEARCH_RESULTS_PER_QUERY": 3},
        )
        gptr_params = {"report_source": "web", "tone": "Objective"}
        chain = gpt_researcher_chain().with_config(
            {"configurable": {"logger": None, "gptr_conf": researcher_conf, "gptr_params": gptr_params}}
        )
        return await chain.ainvoke(query)

    result = asyncio.run(main())
    debug(result)
