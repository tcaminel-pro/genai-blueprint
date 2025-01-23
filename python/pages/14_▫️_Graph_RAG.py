import kuzu
import streamlit as st
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_kuzu.chains.graph_qa.kuzu import KuzuQAChain
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph
from st_cytoscape import cytoscape

from python.ai_core.llm import get_llm
from python.ui_components.cypher_graph_display import get_cytoscape_json, get_cytoscape_style

llm = get_llm(llm_id="gpt_4omini_openai")

st.set_page_config(page_title="Graph RAG with Kuzu", layout="wide")
st.title("Graph RAG with Kuzu")

# Initialize session state for graph data
if "graph" not in st.session_state:
    st.session_state.graph = None

# Form for text input and processing
with st.form("graph_input_form"):
    text = st.text_area(
        "Enter text to analyze:",
        value="Tim Cook is the CEO of Apple. Apple has its headquarters in California.",
        height=100,
    )
    use_cache = st.checkbox("Cache built graph", value=True)
    submitted = st.form_submit_button("Process Text")

DB = "test_db"
if submitted:
    # Define schema
    allowed_nodes = ["Person", "Company", "Location"]
    allowed_relationships = [
        ("Person", "IS_CEO_OF", "Company"),
        ("Company", "HAS_HEADQUARTERS_IN", "Location"),
    ]

    llm_transformer = LLMGraphTransformer(
        llm=llm,
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
        graph_documents,  # type: ignore
        include_source=True,
    )
    st.session_state.graph = graph

# Display the graph if we have data
if graph := st.session_state.graph:
    with st.expander("Graph Visualization", expanded=True):
        cytoscape(
            elements=get_cytoscape_json(graph),  # type: ignore
            stylesheet=get_cytoscape_style(),
            layout={"name": "cose", "animate": True},
            height="600px",
            key="graph_visualization",
        )

with st.form("graph_query_form"):
    st.subheader("Query the Knowledge Graph")
    input = st.text_area(
        "Enter query:",
        value="Tim Cook is the CEO of Apple. Apple has its headquarters in California.",
        height=100,
    )
    submitted = st.form_submit_button("Ask KG!")
    if input and submitted:
        db = kuzu.Database(DB)
        graph = KuzuGraph(db, allow_dangerous_requests=True)

        chain = KuzuQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
        )

        result = chain.invoke({"query": input})
        st.write(f"**Result:** {result['result']}")
