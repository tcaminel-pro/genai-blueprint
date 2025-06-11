"""Deep Search Agent using GPT Researcher.

This module provides a Streamlit interface for configuring and running
GPT Researcher searches with customizable parameters and configuration management.
"""

import asyncio
import textwrap
from collections import deque
from typing import Any, Final

import pandas as pd
import streamlit as st
from devtools import debug
from langchain.callbacks import tracing_v2_enabled
from md2pdf import md2pdf
from streamlit import session_state as sss

from src.ai_core.llm import configurable
from src.ai_extra.gpt_researcher_chain import (
    GptrConfig,
    GptrConfVariables,
    ReportType,
    SearchEngine,
    Tone,
    gpt_researcher_chain,
)
from src.utils.streamlit.auto_scroll import scroll_to_here

LOG_SIZE_MAX = 100

# Initialize session state for configuration
if "current_config" not in sss:
    sss.current_config = "default"

if "custom_config" not in sss:
    sss.custom_config = {}

# Load available configurations and ensure default config exists
available_configs = GptrConfig.list_configs()


st.title("GPT Researcher Playground")

SAMPLE_SEARCH = [
    "What are the ethical issues with AI autonomous agents ? ",
    "What is the architecture of SmolAgents and how it compare with LangGraph ? ",
    "What are the Agentic AI  solutions announced by AWS, Google, Microsoft, SalesForce, Service Now, UI Path, SAP, and other major software editors",
    "Define what is Agentic AI",
]

# Configuration selection
selected_config = st.selectbox(
    "Select Configuration",
    options=available_configs,
    key="selected_config",
    on_change=lambda: sss.update({"custom_config": GptrConfig.load(st.session_state.selected_config).config}),
)
loaded_config = GptrConfig.load(selected_config)
debug(loaded_config)
if not sss.custom_config or sss.custom_config != loaded_config.config:
    sss.custom_config = loaded_config.config.copy()  # Use copy to avoid reference issues

debug(selected_config, sss.custom_config)

# Main configuration area
with st.expander("Search configuration (partial)", expanded=True):
    col1, col2, col3 = st.columns(3)

    # Get current values from session state or defaults
    max_iterations = sss.custom_config.get("max_iterations", 3)
    max_search_results = sss.custom_config.get("max_search_results_per_query", 5)
    report_type = sss.custom_config.get("report_type", "research_report")
    search_engine = sss.custom_config.get("search_engine", "tavily")
    tone = sss.custom_config.get("tone", "Objective")
    llm_id = sss.custom_config.get("llm_id", "gpt_41mini_openrouter")

    # Configuration inputs
    new_max_iterations = col1.number_input("Max Iterations", 1, 5, value=max_iterations)
    new_max_search_results = col1.number_input("Max search per query", 1, 10, value=max_search_results)
    new_llm_id = col1.text_input("LLM ID", value=llm_id)

    new_report_type = col2.selectbox(
        "Report Type",
        [rt.value for rt in ReportType],
        index=[rt.value for rt in ReportType].index(report_type)
        if report_type in [rt.value for rt in ReportType]
        else 0,
    )
    new_search_engine = col2.selectbox(
        "Search Engine",
        [rt.value for rt in SearchEngine],
        index=[rt.value for rt in SearchEngine].index(search_engine)
        if search_engine in [rt.value for rt in SearchEngine]
        else 0,
    )
    new_tone = col2.selectbox(
        "Tone",
        [rt.value for rt in Tone],
        index=[rt.value for rt in Tone].index(tone) if tone in [rt.value for rt in Tone] else 0,
    )

    # Custom prompt for custom report type
    custom_prompt = ""
    if new_report_type == "custom_report":
        custom_prompt = col3.text_area("System prompt:", height=150, value=sss.custom_config.get("custom_prompt", ""))

    # Update session state with new values
    sss.custom_config = {
        "max_iterations": new_max_iterations,
        "max_search_results_per_query": new_max_search_results,
        "report_type": new_report_type,
        "search_engine": new_search_engine,
        "tone": new_tone,
        "llm_id": new_llm_id,
    }

    if custom_prompt:
        sss.custom_config["custom_prompt"] = custom_prompt

# How it works popover
with st.popover("How it works", use_container_width=False):
    c21, c22 = st.columns([7, 12])
    c21.write("Normal Research")
    c21.image(
        "https://private-user-images.githubusercontent.com/13554167/333804350-4ac896fd-63ab-4b77-9688-ff62aafcc527.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDY2NDgzODEsIm5iZiI6MTc0NjY0ODA4MSwicGF0aCI6Ii8xMzU1NDE2Ny8zMzM4MDQzNTAtNGFjODk2ZmQtNjNhYi00Yjc3LTk2ODgtZmY2MmFhZmNjNTI3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTA3VDIwMDEyMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTgwM2RmODQwNjVmZTU0MjI4YTljODJjMzgxY2U1N2MwOGZjNWEyOGM3OTM5ZjNmNmEzMDEwYTg0ZjE5YzllYzUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.SsI3BqipPrR8mL8soiWF0mlCbnXxNnzTip6F4wgY9aM"
    )
    c22.write("Deep Research")
    c22.image("https://github.com/user-attachments/assets/eba2d94b-bef3-4f8d-bbc0-f15bd0a40968")

# Query input area
col1, col2 = st.columns([4, 1])
sample_search = col1.selectbox("Sample queries:", SAMPLE_SEARCH, index=None)


if "log_entries" not in sss:
    sss.log_entries = deque(maxlen=100)
if "research_full_report" not in sss:
    sss.research_full_report = None

if "traces" not in sss:
    sss.traces = {}


class CustomLogsHandler:
    """Handles real-time logging display in Streamlit UI

    Manages both synchronous and asynchronous logging to a Streamlit container.
    Maintains a circular buffer of log entries for display and history.

    Attributes:
        log_container: Streamlit container element for displaying logs
    """

    def __init__(self, log_container, height=200) -> None:
        self.log_container = log_container.container(height=height)

    async def send_json(self, data: dict[str, Any]) -> None:
        debug(data)
        if "log_entries" not in sss:
            sss.log_entries = deque(maxlen=100)
        type = data.get("type")
        if type == "logs":
            sss.log_entries.append(data)
            with self.log_container:
                line = textwrap.shorten(data["output"], 120)
                st.text(line)
                scroll_to_here()

    async def write_log(self, line: str) -> None:
        """Write a single log line to the streamlit container"""
        await self.send_json({"type": "logs", "output": line})


# log_container = None


# This section is no longer needed as we're using the configuration from session state


async def main() -> None:
    """Main async function handling the Streamlit UI and search operations

    Manages:
    - UI layout and state initialization
    - LLM search execution and results display
    - Web research execution and comprehensive reporting
    - Traceability and debugging support

    UI Flow:
    1. User inputs question
    2. Chooses between LLM or Web search
    3. Results displayed in organized tabs:
       - LLM Search: Breakdowns, synthesis, stats
       - Web Search: Report, context, images, sources
    """

    with st.form("my_form"):
        search_input = st.text_area(
            "Your query", height=70, placeholder=" query here...", value=sample_search, label_visibility="collapsed"
        )
        use_cached_result = col2.checkbox(
            "Use cache", value=True, help="Use previous cached search and analysis outcomes "
        )

        submitted_web_search = st.form_submit_button("Web Search", disabled=search_input is None)

        if submitted_web_search and search_input:
            log, report_tab, context_tab, image_tab, sources_tab, stats_tab_web = st.tabs(
                ["log", "**Report**", "Context", "Images", "Sources", "Stats"]
            )
            log_handler = CustomLogsHandler(log, 200)

            # Create researcher configuration from the current settings
            config = sss.custom_config

            # Create GptrConfVariables from the configuration
            researcher_conf = GptrConfVariables(
                fast_llm_id=config.get("llm_id"),
                smart_llm_id=config.get("llm_id"),
                extra_params={
                    "MAX_ITERATIONS": config.get("max_iterations", 3),
                    "MAX_SEARCH_RESULTS_PER_QUERY": config.get("max_search_results_per_query", 5),
                },
            )

            # Set up parameters for the researcher
            gptr_params = {
                "report_source": "web",
                "tone": config.get("tone", "Objective"),
                "report_type": config.get("report_type", "research_report"),
            }

            # Add custom prompt if specified
            if config.get("report_type") == "custom_report" and config.get("custom_prompt"):
                gptr_params["custom_prompt"] = config.get("custom_prompt")

            gptr_chain = gpt_researcher_chain().with_config(
                configurable(
                    {
                        "logger": log_handler,
                        "gptr_conf": researcher_conf,
                        "gptr_params": gptr_params,
                        "use_cached_result": use_cached_result,
                    }
                )
            )
            with tracing_v2_enabled() as cb:
                with st.spinner(text="GPT Researcher running..."):
                    sss.research_full_report = await gptr_chain.ainvoke(search_input)
                    sss.traces["web_search"] = cb.get_run_url()
                    await log_handler.write_log("🍾 The search report is ready !")

            research_full_report = sss.research_full_report
            if research_full_report:
                # write in fist tabs
                sss.web_research_result = research_full_report.report
                report_tab.write(sss.web_research_result)
                context_tab.write(research_full_report.context)

                # 'Image' tab content
                nb_col: Final = 4
                image_tab.write(f"Found images (len: {len(research_full_report.images)})")
                image_cols = image_tab.columns(nb_col)
                for index, image_path in enumerate(research_full_report.images):
                    with image_cols[index % nb_col]:
                        try:
                            st.image(image_path, width=200, caption=f"Image {index + 1}", use_container_width=False)
                        except Exception:
                            st.write(f"cannot display {image_path}")

                # 'Source' tab content
                source_dict = {(s["url"], s["title"]) for s in research_full_report.sources}
                df = pd.DataFrame(source_dict, columns=["url", "title"], index=None)
                sources_tab.dataframe(df)

                # 'Stats' tab content
                stats_tab_web.write(f"Research costs: ${research_full_report.costs}")
                if trace_url := sss.traces.get("web_search"):
                    stats_tab_web.write(f"trace: {trace_url}")

    # Download the PDF file instead of saving it to disk AI!
    if "web_research_result" in sss:
        filename = st.text_input("PDF file name", value="research_report.pdf")
        if st.button("Export to PDF"):
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            try:
                md2pdf(
                    filename,
                    md_content=sss.web_research_result,
                    base_url=None,
                    css_file_path=None,
                )
                st.success(f"Successfully saved PDF to {filename}")
            except Exception as e:
                st.error(f"Error saving PDF: {str(e)}")


asyncio.run(main())
