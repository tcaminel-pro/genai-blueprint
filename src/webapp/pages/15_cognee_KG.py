"""Cognee demonstration page with file upload and knowledge graph processing."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cognee
import streamlit as st
import yaml
from cognee.api.v1.search import SearchType
from loguru import logger
from pydantic import BaseModel
from streamlit import session_state as sss

from src.ai_extra.cognee_utils import set_cognee_config
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

        # Get the HTML content directly
        html_content = await visualize_graph(None)

        # Use components.html to render the HTML with JavaScript
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

    # Choose between file upload or demo
    if not sss.processing_complete:
        option = st.radio(
            "Choose input method:",
            ["Upload Files", "Use Demo"],
            horizontal=True,
            key="input_option",
        )
        if option == "Upload Files":
            uploaded_files = st.file_uploader(
                "Choose files:",
                accept_multiple_files=True,
                type=["txt", "pdf", "docx", "md", "json"],
                key="file_uploader_main",
            )
            if uploaded_files and st.button("🚀 Cognify !", type="primary"):
                with st.spinner("Processing files through cognee pipeline..."):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            success = await process_files([Path(f.name) for f in uploaded_files])
                        else:
                            success = asyncio.run(process_files([Path(f.name) for f in uploaded_files]))
                    except RuntimeError:
                        success = asyncio.run(process_files([Path(f.name) for f in uploaded_files]))
                    if success:
                        sss.processing_complete = True
                        sss.graph_generated = True
                        st.success("✅ Knowledge graph generated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process files")
        elif option == "Use Demo":
            if cognee_demos:
                demo_names = [demo.name for demo in cognee_demos]
                selected_demo_name = st.selectbox("Select a demo:", demo_names, key="demo_select")
                selected_demo = next(d for d in cognee_demos if d.name == selected_demo_name)

                st.write("**Demo texts:**")
                tabs = st.tabs([f"Text {idx}" for idx in range(1, len(selected_demo.texts) + 1)])
                for idx, text in enumerate(selected_demo.texts):
                    with tabs[idx]:
                        st.text_area("", value=text, height=150, key=f"demo_text_{idx}", disabled=True)

                if st.button("🚀 Cognify Demo !", type="primary"):
                    with st.spinner("Processing demo texts through cognee pipeline..."):
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                await cognee.add(data=selected_demo.texts)
                                await cognee.cognify()
                            else:
                                asyncio.run(cognee.add(data=selected_demo.texts))
                                asyncio.run(cognee.cognify())
                        except RuntimeError:
                            asyncio.run(cognee.add(data=selected_demo.texts))
                            asyncio.run(cognee.cognify())
                        sss.processing_complete = True
                        sss.graph_generated = True
                        st.success("✅ Demo knowledge graph generated successfully!")
                        st.rerun()
            else:
                st.warning("No demos found in config/demos/cognee_kg.yaml")

    # Show two-column layout after processing
    if sss.processing_complete:
        st.success("✅ Knowledge graph generated! You can now query and explore the data.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("🔍 Query Knowledge Graph")

            # Build list of suggested queries
            suggested_queries = []
            if sss.selected_demo and cognee_demos:
                demo = next(d for d in cognee_demos if d.name == sss.selected_demo)
                suggested_queries.extend(demo.example_queries)
            if not suggested_queries:
                suggested_queries = [
                    "Who has experience in design tools?",
                    "Summarize key insights",
                    "List main topics",
                ]

            query = st.text_area(
                "Enter your query:",
                placeholder="What insights can you find in these documents?",
                height=100,
                key="query_input",
            )

            search_type = st.selectbox(
                "Search Type:",
                options=[
                    ("Insights", SearchType.INSIGHTS),
                    ("Graph Completion", SearchType.GRAPH_COMPLETION),
                    ("RAG Completion", SearchType.RAG_COMPLETION),
                ],
                format_func=lambda x: x[0],
                key="search_type_select",
            )

            st.write("**Suggested queries:**")
            for q in suggested_queries:
                if st.button(q, key=f"suggest_{q}"):
                    query = q

            if st.button("Search", type="secondary"):
                if query:
                    with st.spinner("Searching knowledge graph..."):
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                results = await cognee.search(query_type=search_type[1], query_text=query)
                            else:
                                results = asyncio.run(cognee.search(query_type=search_type[1], query_text=query))
                        except RuntimeError:
                            results = asyncio.run(cognee.search(query_type=search_type[1], query_text=query))
                        if (
                            isinstance(results, list)
                            and results
                            and isinstance(results[0], str)
                            and "error" in results[0]
                        ):
                            st.error(f"Error: {results[0]}")
                        else:
                            st.success("Results found!")
                            st.json(results)
                else:
                    st.warning("Please enter a query")

        with col2:
            st.header("🕸️ Knowledge Graph Visualization")
            await display_graph_visualization()


if __name__ == "__main__":
    # Handle both direct execution and Streamlit execution
    try:
        # Check if we're in a running event loop (Streamlit)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a task for async execution in Streamlit
            loop.create_task(main())
        else:
            # Run normally when not in Streamlit
            asyncio.run(main())
    except RuntimeError:
        # Fallback for when there's no event loop
        asyncio.run(main())
