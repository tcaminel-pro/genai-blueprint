""" """

import asyncio
import textwrap
from collections import deque
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from langchain.callbacks import tracing_v2_enabled

from python.ai_core.llm import configurable
from python.ai_extra.gpt_researcher_chain import GptrConf, gpt_researcher_chain

LOG_SIZE_MAX = 100

GPTR_LLM_ID = "gpt_4omini_openrouter"
CUSTOM_GPTR_CONFIG = {
    "MAX_ITERATIONS": 3,
    "MAX_SEARCH_RESULTS_PER_QUERY": 5,
}

st.title("GPT Researcher Playground")


SAMPLE_SEARCH = ["What are the ethical issues with AI autonomous agents ? "]

col1, co2 = st.columns([4, 1])
sample_search = col1.selectbox("Sample queries", SAMPLE_SEARCH, index=None)

search_input = col1.text_area("Your query", height=70, placeholder=" query here...", value=sample_search)
use_cached_result = co2.checkbox("Use cache", value=True, help="Use previous cached search and analysis outcomes ")


if "log_entries" not in st.session_state:
    st.session_state.log_entries = deque(maxlen=100)
if "research_full_report" not in st.session_state:
    st.session_state.research_full_report = None

if "traces" not in st.session_state:
    st.session_state.traces = dict()


class CustomLogsHandler:
    """Handles real-time logging display in Streamlit UI

    Manages both synchronous and asynchronous logging to a Streamlit container.
    Maintains a circular buffer of log entries for display and history.

    Attributes:
        log_container: Streamlit container element for displaying logs
    """

    def __init__(self, log_container, height=200):
        self.log_container = log_container.container(height=height)

    async def send_json(self, data: dict[str, Any]) -> None:
        if "log_entries" not in st.session_state:
            st.session_state.log_entries = deque(maxlen=100)
        data["ts"] = datetime.now().strftime("%H:%M:%S")
        st.session_state.log_entries.append(data)

        # Add autoscrolling so the list line is always displayed AI!
        with self.log_container:

            def stream_log():
                for entry in st.session_state["log_entries"]:
                    line = textwrap.shorten(entry["output"], 120)
                    yield f"{line}\n"

            st.write_stream(stream_log)

    async def write_log(self, line: str):
        """Write a single log line to the streamlit container"""
        await self.send_json({"output": line})


# log_container = None


researcher_conf = GptrConf(
    # fast_llm_id=gpt_llm,
    # smart_llm_id=gpt_llm,
    # strategic_llm_id=gpt_llm,
    extra_params=CUSTOM_GPTR_CONFIG
)


async def main():
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
        submitted_web_search = st.form_submit_button("Web Search", disabled=search_input is None)

        if submitted_web_search and search_input:
            log, report_tab, context_tab, image_tab, sources_tab, stats_tab_web = st.tabs(
                ["log", "**Report**", "Context", "Images", "Sources", "Stats"]
            )
            log_handler = CustomLogsHandler(log, 200)

            gptr_params = {"report_source": "web", "tone": "Objective"}
            gptr_chain = gpt_researcher_chain().with_config(
                configurable(
                    {
                        "logger": log_handler,
                        "gptr_conf": researcher_conf,
                        "gptr_params": gptr_params,
                        "use_cached_result": True,
                    }
                )
            )
            with tracing_v2_enabled() as cb:
                with st.spinner(text="searching the web..."):
                    st.session_state.research_full_report = await gptr_chain.ainvoke(search_input)
                    st.session_state.traces["web_search"] = cb.get_run_url()
                    await log_handler.write_log("The search report ready !")

            research_full_report = st.session_state.research_full_report
            if research_full_report:
                # write in fist tabs
                web_research_result = research_full_report.report
                report_tab.write(web_research_result)
                context_tab.write(research_full_report.context)

                # 'Image' tab content
                NB_COL = 4
                image_tab.write(f"Found images (len: {len(research_full_report.images)})")
                image_cols = image_tab.columns(NB_COL)
                for index, image_path in enumerate(research_full_report.images):
                    with image_cols[index % NB_COL]:
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
                if trace_url := st.session_state.traces.get("web_search"):
                    stats_tab_web.write(f"trace: {trace_url}")


asyncio.run(main())
