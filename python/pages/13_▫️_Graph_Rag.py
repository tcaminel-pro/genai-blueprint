import streamlit as st
from pathlib import Path
import kuzu
from st_cytoscape import cytoscape
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_kuzu.chains.graph_qa.kuzu import KuzuQAChain
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Graph RAG with Kuzu", layout="wide")
st.title("Graph RAG with Kuzu")

# Initialize session state for graph data
if "cyto_data" not in st.session_state:
    st.session_state.cyto_data = {"nodes": [], "edges": []}

# Form for text input and processing
with st.form("graph_input_form"):
    text = st.text_area(
        "Enter text to analyze:",
        value="Tim Cook is the CEO of Apple. Apple has its headquarters in California.",
        height=100,
    )
    use_cache = st.checkbox("Cache built graph", value=True)
    submitted = st.form_submit_button("Process Text")

if submitted:
    DB = "test_db"
    
    # Clear existing database if not using cache
    if not use_cache and Path(DB).exists():
        import shutil
        shutil.rmtree(DB)

    if not Path(DB).exists():
        # Define schema
        allowed_nodes = ["Person", "Company", "Location"]
        allowed_relationships = [
            ("Person", "IS_CEO_OF", "Company"),
            ("Company", "HAS_HEADQUARTERS_IN", "Location"),
        ]
        
        # Define the LLMGraphTransformer
        llm_transformer = LLMGraphTransformer(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
        )

        # Convert the given text into graph documents
        documents = [Document(page_content=text)]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        db = kuzu.Database(DB)
        graph = KuzuGraph(db, allow_dangerous_requests=True)

        # Add the graph document to the graph
        graph.add_graph_documents(
            graph_documents,
            include_source=True,
        )

        # Prepare cytoscape data
        st.session_state.cyto_data = {"nodes": [], "edges": []}

        # Query all nodes
        nodes_query = "MATCH (n) RETURN n"
        nodes_result = graph.query(nodes_query)

        # Add nodes to Cytoscape data, excluding text_chunk nodes
        for node in nodes_result:
            if node["n"]["type"] != "text_chunk":
                st.session_state.cyto_data["nodes"].append(
                    {"data": {"id": node["n"]["id"], "label": node["n"]["_label"], "type": node["n"]["type"]}}
                )

        # Query all relationships
        rels_query = "MATCH (a)-[r]->(b) RETURN a, r, b"
        rels_result = graph.query(rels_query)

        # Add relationships to Cytoscape data, excluding those involving text_chunk nodes
        for rel in rels_result:
            if rel["a"]["type"] != "text_chunk" and rel["b"]["type"] != "text_chunk":
                st.session_state.cyto_data["edges"].append(
                    {
                        "data": {
                            "id": f"{rel['a']['id']}-{rel['r']['_label']}-{rel['b']['id']}",
                            "source": rel["a"]["id"],
                            "target": rel["b"]["id"],
                            "label": rel["r"]["_label"],
                        }
                    }
                )

# Display the graph if we have data
if st.session_state.cyto_data["nodes"]:
    st.subheader("Knowledge Graph Visualization")
    cytoscape(
        elements=st.session_state.cyto_data["nodes"] + st.session_state.cyto_data["edges"],
        stylesheet=[
            {
                "selector": "node",
                "style": {
                    "content": "data(id)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "color": "white",
                    "background-color": "#002733",
                    "width": "label",
                    "height": "label",
                    "padding": "10px",
                    "shape": "ellipse",
                    "font-size": "12px",
                    "border-width": 2,
                    "border-color": "#444",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "content": "data(label)",
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "line-color": "#999",
                    "target-arrow-color": "#999",
                    "font-size": "6px",
                    "text-margin-y": "-10px",
                    "text-rotation": "autorotate",
                    "width": 2,
                    "arrow-scale": 1,
                },
            },
        ],
        layout={"name": "cose", "animate": True},
        height="600px",
        key="graph_visualization",
    )

    # Query section
    st.subheader("Query the Knowledge Graph")
    query = st.text_input("Enter your query:")
    if query:
        db = kuzu.Database(DB)
        graph = KuzuGraph(db, allow_dangerous_requests=True)
        
        chain = KuzuQAChain.from_llm(
            llm=ChatOpenAI(model="gpt-4", temperature=0.3),
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
        )
        
        result = chain.invoke(query)
        st.write(f"**Query:** {query}")
        st.write(f"**Result:** {result['result']}")
