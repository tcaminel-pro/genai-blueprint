"""Cognee demonstration page with file upload and knowledge graph processing."""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from cognee import cognify
from cognee.api.v1.search import SearchType
from loguru import logger
from streamlit import session_state as sss

from src.ai_extra.cognee_utils import set_cognee_config
from utils.logger_factory import setup_logging

setup_logging()
logger = logger


async def process_files(uploaded_files: List[Any]) -> bool:
    """Process uploaded files through cognee pipeline."""
    if not uploaded_files:
        return False

    try:
        # Save uploaded files to temporary directory
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(str(file_path))

        # Run cognify pipeline
        await cognify(file_paths)

        # Clean up temporary files
        for file_path in file_paths:
            os.remove(file_path)

        return True
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return False


async def query_knowledge_graph(query: str, search_type: SearchType) -> Dict[str, Any]:
    """Query the knowledge graph with specified search type."""
    try:
        if search_type == SearchType.INSIGHTS:
            results = await graph_search(query_type=SearchType.INSIGHTS, query=query)
        elif search_type == SearchType.GRAPH_COMPLETION:
            results = await graph_search(query_type=SearchType.GRAPH_COMPLETION, query=query)
        elif search_type == SearchType.RAG_COMPLETION:
            results = await rag_search(query=query)
        else:
            results = {"error": "Invalid search type"}

        return results
    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}")
        return {"error": str(e)}


def display_graph_visualization():
    """Display the knowledge graph visualization."""
    try:
        if display_graph is not None:
            graph_html = display_graph()
            st.components.v1.html(graph_html, height=600, scrolling=True)
        else:
            st.error("Graph visualization not available")
    except Exception as e:
        st.error(f"Error displaying graph: {e}")


def main():
    """Main Streamlit page for Cognee demonstration."""
    st.set_page_config(page_title="Cognee Demo", page_icon="🧠", layout="wide")

    st.title("🧠 Cognee Knowledge Graph Demo")
    st.markdown("Upload documents and explore knowledge graph insights")

    # Initialize cognee
    set_cognee_config()

    # Initialize session state
    if "processing_complete" not in sss:
        sss.processing_complete = False
    if "graph_generated" not in sss:
        sss.graph_generated = False
    if "show_upload_popup" not in sss:
        sss.show_upload_popup = False

    # Show upload popup if not processed yet
    if not sss.processing_complete and not sss.show_upload_popup:
        if st.button("📁 Select Files to Upload and Process", type="primary", use_container_width=True):
            sss.show_upload_popup = True
            st.rerun()

    # Handle file upload popup
    if sss.show_upload_popup and not sss.processing_complete:
        with st.popover("📁 Upload and Process Files", use_container_width=True):
            st.header("📁 File Upload")

            uploaded_files = st.file_uploader(
                "Choose files to process",
                accept_multiple_files=True,
                type=["txt", "pdf", "docx", "md", "json"],
                key="file_uploader_popup",
            )

            if uploaded_files:
                st.info(f"Uploaded {len(uploaded_files)} file(s)")

                if st.button("🚀 Start Cognify Pipeline", type="primary"):
                    with st.spinner("Processing files through cognee pipeline..."):
                        success = asyncio.run(process_files(uploaded_files))

                        if success:
                            sss.processing_complete = True
                            sss.graph_generated = True
                            sss.show_upload_popup = False
                            st.success("✅ Knowledge graph generated successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to process files")

    # Show two-column layout after processing
    if sss.processing_complete:
        st.success("✅ Knowledge graph generated! You can now query and explore the data.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("🔍 Query Knowledge Graph")

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

            if st.button("Search", type="secondary"):
                if query:
                    with st.spinner("Searching knowledge graph..."):
                        results = asyncio.run(query_knowledge_graph(query, search_type[1]))

                        if "error" in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.success("Results found!")
                            st.json(results)
                else:
                    st.warning("Please enter a query")

        with col2:
            st.header("📊 Knowledge Graph Visualization")
            display_graph_visualization()


if __name__ == "__main__":
    main()
