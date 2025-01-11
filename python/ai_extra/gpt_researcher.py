"""
GPT Researcher integration utilities.

This module provides classes and functions to manage and configure GPT Researcher's
language models, settings, and research operations. It handles:

- Configuration of LLM models and embeddings
- Creation of temporary configuration files
- Execution of research tasks with configurable parameters
- Management of research reports and metadata

The main classes are:
- GptResearcherConf: Configuration management
- ResearchReport: Container for research results
"""

import json
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from python.ai_core.llm import LlmFactory

load_dotenv()


class GptResearcherConf(BaseModel):
    """Configuration class for GPT Researcher's language models and settings.

    Attributes:
        fast_llm_id (str | None): Identifier for the fast language model.
        smart_llm_id (str | None): Identifier for the smart language model.
        strategic_llm_id (str | None): Identifier for the strategic language model.
        embeddings_id (str | None): Identifier for the embeddings model (not yet implemented).
        extra_params (dict): Additional configuration parameters to include.

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

        Returns:
            str: Path to the temporary configuration file.

        Raises:
            NotImplementedError: If embeddings configuration is attempted (not yet supported).

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

        path = Path(tempfile.gettempdir()) / "gpt_researcher_conf.json"
        with open(path, "w") as json_file:
            json.dump(config_dict, json_file)

        debug(config_dict)

        # logger.debug("GptResearcherConfig:", config_dict)
        return str(path)


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
    query: str,
    role: str,
    researcher_config: GptResearcherConf,
    report_type: str = "custom_report",
    report_source: str = "web",
    verbose: bool = True,
    logger: Optional[Any] = None,
) -> ResearchReport:
    """Execute a GPT Researcher task with configurable parameters.

    Args:
        query: Research query/prompt
        role: Role description for the researcher
        researcher_config: Configuration for LLM models
        report_type: Type of report to generate
        report_source: Source for research ('web' or 'arxiv')
        verbose: Enable verbose output
        logger: Optional logger for progress tracking


    Returns:
        ResearchReport: Container with research results and metadata
    """
    from gpt_researcher import GPTResearcher

    researcher = GPTResearcher(
        query=query,
        role=role,
        report_type=report_type,
        report_source=report_source,
        verbose=verbose,
        websocket=logger,
        config_path=researcher_config.get_config_path(),
    )
    _ = await researcher.conduct_research()
    report = await researcher.write_report()

    return ResearchReport(
        report=report,
        context=str(researcher.get_research_context()),
        costs=researcher.get_costs(),
        images=[str(e) for e in researcher.get_research_images()],
        sources=researcher.get_research_sources(),
    )
