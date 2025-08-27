"""Cognee demonstration page with file upload and knowledge graph processing."""

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from loguru import logger

try:
    import cognee
    from cognee import cognify
    from cognee.api.v1.search import SearchType
    from cognee.api.v1.search.graph_search import graph_search
    from cognee.api.v1.search.rag_search import rag_search
    from cognee.modules.graph.visualisation import display_graph
    from cognee.shared.data_models import GraphDBType
    from cognee.base_config import get_base_config
    COGNEE_AVAILABLE = True
except ImportError:
    COGNEE_AVAILABLE = False
    cognee = None
    SearchType = None
    display_graph = None

from utils.logger_factory import setup_logging

setup_logging()
logger = logger


def initialize_cognee():
    """Initialize cognee with appropriate configuration."""
    if not COGNEE_AVAILABLE:
        return False
    
    try:
        # Set up cognee configuration
        base_config = get_base_config()
        base_config.graph_db_url = os.getenv("GRAPH_DB_URL", "bolt://localhost:7687")
        base_config.graph_db_username = os.getenv("GRAPH_DB_USERNAME", "neo4j")
        base_config.graph_db_password = os.getenv("GRAPH_DB_PASSWORD", "password")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize cognee: {e}")
        return False


async def process_files(uploaded_files: List[Any]) -> bool:
    """Process uploaded files through cognee pipeline."""
    if not uploaded_files or not COGNEE_AVAILABLE:
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
    st.set_page_config(
        page_title="Cognee Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Cognee Knowledge Graph Demo")
    st.markdown("Upload documents and explore knowledge graph insights")
    
    if not COGNEE_AVAILABLE:
        st.error("""
        Cognee is not installed or available. 
        Please install with: `pip install cognee`
        """)
        return
    
    # Initialize cognee
    if not initialize_cognee():
        st.error("Failed to initialize cognee configuration")
        return
    
    # Initialize session state
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "graph_generated" not in st.session_state:
        st.session_state.graph_generated = False
    
    # Sidebar for graph visualization
    with st.sidebar:
        st.header("Knowledge Graph")
        
        if st.session_state.graph_generated:
            display_graph_visualization()
        else:
            st.info("Upload and process files to see the knowledge graph")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 File Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files to process",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md', 'json']
        )
        
        if uploaded_files:
            st.info(f"Uploaded {len(uploaded_files)} file(s)")
            
            if st.button("🚀 Start Cognify Pipeline", type="primary"):
                with st.spinner("Processing files through cognee pipeline..."):
                    success = asyncio.run(process_files(uploaded_files))
                    
                    if success:
                        st.session_state.processing_complete = True
                        st.session_state.graph_generated = True
                        st.success("✅ Knowledge graph generated successfully!")
                    else:
                        st.error("❌ Failed to process files")
    
    with col2:
        st.header("🔍 Query Knowledge Graph")
        
        if st.session_state.processing_complete:
            query = st.text_area(
                "Enter your query:",
                placeholder="What insights can you find in these documents?",
                height=100
            )
            
            search_type = st.selectbox(
                "Search Type:",
                options=[
                    ("Insights", SearchType.INSIGHTS),
                    ("Graph Completion", SearchType.GRAPH_COMPLETION),
                    ("RAG Completion", SearchType.RAG_COMPLETION)
                ],
                format_func=lambda x: x[0]
            )
            
            if st.button("Search", type="secondary"):
                if query:
                    with st.spinner("Searching knowledge graph..."):
                        results = asyncio.run(query_knowledge_graph(query, search_type[1]))
                        
                        if "error" in results:
                            st.error(f"Error: {results['error']}")
                        else:
                            st.success("Results found!")
                            
                            # Display results
                            st.json(results)
                else:
                    st.warning("Please enter a query")
        else:
            st.info("Process files first to enable querying")


if __name__ == "__main__":
    main()
