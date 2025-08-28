"""Cognee demonstration page with file upload and knowledge graph processing."""

import asyncio
import tempfile
from pathlib import Path
from typing import List, Sequence

import cognee
import streamlit as st
from beartype.door import infer_hint, is_bearable
from cognee.api.v1.search import SearchType
from devtools import debug  # noqa: F401
from loguru import logger
from pydantic import BaseModel
from streamlit import session_state as sss
from streamlit.delta_generator import DeltaGenerator

from src.ai_extra.cognee_utils import get_search_type_description, set_cognee_config
from src.utils.config_mngr import global_config
from utils.logger_factory import setup_logging

setup_logging()
logger = logger


async def process_files(uploaded_files: Sequence[Path]) -> bool:
    """Process uploaded files through cognee pipeline."""
    if not uploaded_files:
        return False

    try:
        files = [str(f) for f in uploaded_files]
        await cognee.add(data=files)
        await cognee.cognify()

    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return False
    return True


async def query_knowledge_graph(query: str, search_type: SearchType) -> list:
    """Query the knowledge graph with specified search type."""
    try:
        results = await cognee.search(query_type=SearchType.INSIGHTS, query_text=query)
        return results
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}")
        return [f"error: {e}"]


async def display_graph_visualization():
    """Display the knowledge graph visualization."""
    try:
        import streamlit.components.v1 as components
        from cognee.api.v1.visualize.visualize import visualize_graph

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            html_content = await visualize_graph(tmp_path)
            components.html(html_content, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Error displaying graph: {e}")


class CogneeDemo(BaseModel):
    """Configuration for a Cognee demo preset."""

    name: str
    texts: List[str]
    example_queries: List[str]


def load_demos_from_config() -> List[CogneeDemo]:
    """Load demo configurations from global config."""
    try:
        demos_config_path = "config/demos/cognee_kg.yaml"
        config = global_config().merge_with(demos_config_path).get_dict("cognee-demo")
        demos = []
        for demo_name, demo_data in config.items():
            texts = demo_data.get("texts", [])
            example_queries = demo_data.get("example_queries", [])
            demos.append(CogneeDemo(name=demo_name, texts=texts, example_queries=example_queries))
        return demos
    except Exception as e:
        logger.error(f"Error loading demos from config: {e}")
        return []


cognee_demos = load_demos_from_config()


async def _render_input_section():
    """Render the input section for file upload or demo selection."""
    TEXT1 = "Upload Files"
    TEXT2 = "Predefined demo"
    option = st.radio(
        "Choose input method:",
        [TEXT1, TEXT2],
        horizontal=True,
        key="input_option",
    )
    if option == TEXT1:
        await _handle_file_upload()
    elif option == TEXT2:
        await _handle_demo_selection()


async def _handle_file_upload():
    """Handle file upload and processing."""
    uploaded_files = st.file_uploader(
        "Choose files:",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "md", "json"],
        key="file_uploader_main",
    )
    if uploaded_files and st.button("🚀 Cognify !", type="primary"):
        with st.spinner("Processing files through cognee pipeline..."):
            success = await process_files([Path(f.name) for f in uploaded_files])
            if success:
                sss.processing_complete = True
                sss.graph_generated = True
                st.success("✅ Knowledge graph generated successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to process files")


async def _handle_demo_selection():
    """Handle demo selection and processing."""
    if not cognee_demos:
        st.warning("No demos found in config/demos/cognee_kg.yaml")
        return

    demo_names = [demo.name for demo in cognee_demos]
    selected_demo_name = st.selectbox("Select a demo:", demo_names, key="demo_select")
    selected_demo = next(d for d in cognee_demos if d.name == selected_demo_name)
    sss.selected_demo = selected_demo

    st.write("**Demo texts:**")
    tabs = st.tabs([f"Text {idx}" for idx in range(1, len(selected_demo.texts) + 1)])
    for idx, text in enumerate(selected_demo.texts):
        with tabs[idx]:
            st.text_area("", value=text, height=150, key=f"demo_text_{idx}", disabled=True)

    if st.button("🚀 Cognify Demo !", type="primary"):
        with st.spinner("Processing demo texts through cognee pipeline..."):
            await cognee.add(data=selected_demo.texts)
            await cognee.cognify()
            sss.processing_complete = True
            sss.graph_generated = True
            st.success("✅ Demo knowledge graph generated successfully!")
            st.rerun()


async def _render_results_section():
    """Render the results section with query and visualization."""
    st.success("✅ Knowledge graph generated! You can now query and explore the data.")

    col1, col2 = st.columns([1, 1])

    with col1:
        await _render_query_section()
    with col2:
        st.header("🕸️ Knowledge Graph Visualization")
        await display_graph_visualization()


def _display_input_form(w: DeltaGenerator, suggested_queries: list[str]) -> tuple[str, bool]:
    """Displays the input form and returns user input."""
    # Move selectbox outside form so it updates immediately
    sample_search = st.selectbox(
        label="Sample queries",
        placeholder="Select an example (optional)",
        options=suggested_queries,
        index=None,
        key="sample_query_select",
    )
    with w.form("my_form", border=False):
        cf1, cf2 = st.columns([15, 1], vertical_alignment="bottom")
        prompt = cf1.text_area(
            "Your task",
            height=68,
            placeholder="Enter or modify your query here...",
            value=sample_search or "",
            label_visibility="collapsed",
        )
        submitted = cf2.form_submit_button(label="", icon=":material/send:")
    return prompt, submitted


async def _render_query_section():
    """Render the query section for knowledge graph exploration."""
    st.header("🔍 Query Knowledge Graph")

    # Build list of suggested queries

    suggested_queries = []
    if sss.selected_demo and cognee_demos:
        suggested_queries.extend(sss.selected_demo.example_queries)

    col1, col2 = st.columns([1, 4])
    search_type = col1.selectbox(
        "Search Type:",
        options=[  # from https://docs.cognee.ai/core-concepts/main-operations/search
            ("Insights", SearchType.INSIGHTS),
            ("RAG Completion", SearchType.RAG_COMPLETION),
            ("Graph Completion", SearchType.GRAPH_COMPLETION),
            ("Summaries", SearchType.SUMMARIES),
            ("Chunks", SearchType.CHUNKS),
            ("Graph Summary Completion", SearchType.GRAPH_SUMMARY_COMPLETION),
            ("Code", SearchType.CODE),
            ("Cypher", SearchType.CYPHER),
            ("Natural Language", SearchType.NATURAL_LANGUAGE),
            ("Graph Completion CoT", SearchType.GRAPH_COMPLETION_COT),
            ("Graph Completion Context Extension", SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION),
            ("Feeling Lucky", SearchType.FEELING_LUCKY),
            ("Feedback", SearchType.FEEDBACK),
        ],
        format_func=lambda x: x[0],
        key="search_type_select",
        label_visibility="collapsed",
        placeholder="Search type",
    )
    # Display the description
    st.caption(f"🔍 {get_search_type_description(search_type[1])}")
    query, submitted = _display_input_form(col2, suggested_queries)

    if submitted:
        if not query:
            st.warning("Please enter a query")
        else:
            with st.spinner("Searching knowledge graph..."):
                try:
                    results = await cognee.search(query_type=search_type[1], query_text=query)
                except Exception as ex:
                    logger.exception(ex)
                    st.error(f"error: {ex}")
                    return

                # Display results
                results_type = infer_hint(results)

                if is_bearable(results, list[dict]) and results:
                    st.success("Results found!")
                    st.dataframe(results)
                elif is_bearable(results, list[str]) and results:
                    st.success("Results found!")
                    for r in results:
                        st.write(r)
                elif is_bearable(results, list[tuple[dict, ...]]) and results:
                    st.success("Results found!")
                    # Flatten list of tuples of dicts and display as dataframe with JSON cells
                    import json

                    flattened = [item for sublist in results for item in sublist]
                    if flattened:
                        # Convert dicts to JSON strings for display
                        json_data = [
                            {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in item.items()}
                            for item in flattened
                        ]
                        st.dataframe(json_data)
                    else:
                        st.info("No data found in nested lists")
                elif is_bearable(results, list[tuple[dict, dict]]) and results:
                    st.success("Results found!")
                    # Handle case where we have tuple pairs of dicts
                    import json

                    flattened = []
                    for pair in results:
                        if len(pair) >= 2:
                            flattened.extend(pair)

                    if flattened:
                        json_data = [
                            {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in item.items()}
                            for item in flattened
                        ]
                        st.dataframe(json_data)
                    else:
                        st.info("No data found in nested structures")
                else:
                    st.success("Results found! ")
                    st.json(results)


async def main():
    """Main Streamlit page for Cognee demonstration."""
    st.set_page_config(page_title="Cognee Demo", page_icon="🧠", layout="wide")

    st.title("🧠 Knowledge Graph Demo")
    st.markdown("Upload documents or use predefined demos to explore knowledge graph insights")

    # Initialize cognee
    set_cognee_config()

    # Initialize session state
    if "processing_complete" not in sss:
        sss.processing_complete = False
    if "graph_generated" not in sss:
        sss.graph_generated = False
    if "show_upload_popup" not in sss:
        sss.show_upload_popup = False
    if "selected_demo" not in sss:
        sss.selected_demo = None

    if not sss.processing_complete:
        await _render_input_section()
    if sss.processing_complete:
        await _render_results_section()


if __name__ == "__main__":
    # Handle both direct execution and Streamlit execution
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            asyncio.run(main())
    except RuntimeError:
        asyncio.run(main())
