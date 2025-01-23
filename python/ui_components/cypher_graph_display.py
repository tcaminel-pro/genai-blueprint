# Add module doc and function docstrings AI!


from langchain_kuzu.graphs.graph_store import GraphStore


def get_cytoscape_json(graph: GraphStore) -> dict:
    # Get all nodes and relationships in Cytoscape JSON format
    cyto_data = {"nodes": [], "edges": []}

    # Query all nodes
    nodes_query = "MATCH (n) RETURN n"
    nodes_result = graph.query(nodes_query)

    # Add nodes to Cytoscape data, excluding text_chunk nodes
    for node in nodes_result:
        if node["n"]["type"] != "text_chunk":
            cyto_data["nodes"].append(
                {"data": {"id": node["n"]["id"], "label": node["n"]["_label"], "type": node["n"]["type"]}}
            )

    # Query all relationships
    rels_query = "MATCH (a)-[r]->(b) RETURN a, r, b"
    rels_result = graph.query(rels_query)

    # Add relationships to Cytoscape data, excluding those involving text_chunk nodes
    for rel in rels_result:
        if rel["a"]["type"] != "text_chunk" and rel["b"]["type"] != "text_chunk":
            cyto_data["edges"].append(
                {
                    "data": {
                        "id": f"{rel['a']['id']}-{rel['r']['_label']}-{rel['b']['id']}",
                        "source": rel["a"]["id"],
                        "target": rel["b"]["id"],
                        "label": rel["r"]["_label"],
                    }
                }
            )
    return cyto_data


def get_cytoscape_style() -> list[dict]:
    style = [
        {
            "selector": "node",
            "css": {
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
            "css": {
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
    ]
    return style
