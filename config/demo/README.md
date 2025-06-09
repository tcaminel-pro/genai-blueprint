# GPT Researcher Configurations

This directory contains YAML configuration files for the GPT Researcher agent.

Each file follows the naming convention `gptr_config_<name>.yaml` and contains:

- `name`: Display name of the configuration
- `description`: Brief description of what this configuration is for
- `config`: Dictionary of configuration parameters including:
  - `max_iterations`: Number of research iterations (1-5)
  - `max_search_results_per_query`: Number of search results per query (1-10)
  - `report_type`: Type of report to generate (research_report, detailed_report, outline_report, custom_report, deep)
  - `search_engine`: Search engine to use (tavily, duckduckgo, google, bing, etc.)
  - `tone`: Tone of the report (Objective, Analytical, Informative, etc.)
  - `llm_id`: ID of the LLM to use
  - `custom_prompt`: Custom system prompt (for custom_report type)
