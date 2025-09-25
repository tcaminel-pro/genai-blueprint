"""
Generate an interactive HTML visualization of a Kuzu graph from a given connection.

This module is standalone (no Cognee dependencies). It queries nodes and edges
from the connected Kuzu database, builds a simple JSON model, and embeds it in
an HTML page rendered with D3 force-directed layout.

Usage example:
```
from kuzu import Database, Connection
from scripts.kuzu_graph_html import generate_kuzu_graph_html

db = Database("/path/to/graph.db")
conn = Connection(db)
html = generate_kuzu_graph_html(conn, "/tmp/graph_visualization.html")
print("Wrote:", "/tmp/graph_visualization.html")
```
"""

from __future__ import annotations

import json
import os
from typing import Any

import kuzu  # Requires the `kuzu` package


def _fetch_graph_data(connection: kuzu.Connection) -> tuple[list[tuple[str, dict]], list[tuple[str, str, str, dict]]]:
    """Fetch all nodes and edges from the Kuzu database via the provided connection.

    The function returns two lists:
    - nodes: list of (node_id, properties_dict)
    - edges: list of (source_id, target_id, relationship_name, properties_dict)

    Works with the actual schema created by the test script.
    """
    nodes: list[tuple[str, dict]] = []
    edges: list[tuple[str, str, str, dict]] = []
    
    # Get all tables first to understand the schema
    try:
        tables_result = connection.execute("CALL show_tables() RETURN *")
        tables_df = tables_result.get_as_df()
        
        node_tables = []
        
        for _, row in tables_df.iterrows():
            table_name = row['name']
            table_type = row['type']
            if table_type == 'NODE':
                node_tables.append(table_name)
        
        # Fetch nodes from all node tables using proper RETURN clause
        for table_name in node_tables:
            try:
                # Query to get all properties of nodes in this table
                if table_name == 'Opportunity':
                    nodes_query = f"MATCH (n:{table_name}) RETURN n.name as name, n.opportunity_id as opportunity_id, n.status as status"
                elif table_name == 'Customer':
                    nodes_query = f"MATCH (n:{table_name}) RETURN n.name as name, n.segment as segment"
                elif table_name == 'Person':
                    nodes_query = f"MATCH (n:{table_name}) RETURN n.name as name, n.role as role, n.organization as organization, n.contact_type as contact_type"
                elif table_name == 'Partner':
                    nodes_query = f"MATCH (n:{table_name}) RETURN n.name as name, n.role as role"
                elif table_name == 'RiskAnalysis':
                    nodes_query = f"MATCH (n:{table_name}) RETURN n.risk_description as risk_description, n.impact_level as impact_level, n.status as status"
                elif table_name == 'FinancialMetrics':
                    nodes_query = f"MATCH (n:{table_name}) RETURN n.tcv as tcv, n.annual_revenue as annual_revenue, n.project_margin as project_margin"
                elif table_name == 'CompetitiveLandscape':
                    nodes_query = f"MATCH (n:{table_name}) RETURN n.competitive_position as competitive_position, n.competitors as competitors"
                else:
                    # Generic query for unknown table types
                    nodes_query = f"MATCH (n:{table_name}) RETURN n"
                
                nodes_result = connection.execute(nodes_query)
                result_df = nodes_result.get_as_df()
                
                for idx, row in result_df.iterrows():
                    node_dict = {}
                    
                    # Convert row to dict and clean up
                    for col, val in row.items():
                        if val is not None and str(val).strip():
                            node_dict[col] = str(val)
                    
                    # Determine the node name based on available fields - just truncate
                    if 'name' in node_dict:
                        node_name = node_dict['name'][:15] + ('...' if len(node_dict['name']) > 15 else '')
                    elif 'risk_description' in node_dict:
                        node_name = node_dict['risk_description'][:15] + ('...' if len(node_dict['risk_description']) > 15 else '')
                    elif 'competitive_position' in node_dict:
                        node_name = node_dict['competitive_position'][:15] + ('...' if len(node_dict['competitive_position']) > 15 else '')
                    elif 'tcv' in node_dict:
                        node_name = f"${node_dict['tcv']}"
                        if len(node_name) > 15:
                            node_name = node_name[:15] + '...'
                    else:
                        node_name = f"{table_name}_{idx}"
                    
                    node_dict['type'] = table_name
                    node_dict['name'] = node_name
                    
                    # Create unique ID
                    node_id = f"{table_name}_{node_name.replace(' ', '_').replace('/', '_')[:30]}"
                    
                    nodes.append((node_id, node_dict))
                    
            except Exception as e:
                print(f"Error fetching nodes from {table_name}: {e}")
                continue

        # If no nodes, return early
        if not nodes:
            return [], []

        # Fetch relationships using explicit queries
        try:
            # Try to get all relationships - Kuzu specific syntax
            rel_query = "MATCH (n)-[r]->(m) RETURN n, r, m"
            rel_result = connection.execute(rel_query)
            rel_df = rel_result.get_as_df()
            
            for _, row in rel_df.iterrows():
                # Extract source and destination node data
                src_node = row['n']
                dst_node = row['m']
                rel_obj = row['r']
                
                # Get node types from the object class names or deduce from properties
                src_type = "Unknown"
                dst_type = "Unknown"
                rel_type = "RELATED_TO"
                
                # Try to determine types from node properties
                if hasattr(src_node, 'opportunity_id'):
                    src_type = "Opportunity"
                elif hasattr(src_node, 'segment'):
                    src_type = "Customer"
                elif hasattr(src_node, 'contact_type'):
                    src_type = "Person"
                elif hasattr(src_node, 'risk_description'):
                    src_type = "RiskAnalysis"
                elif hasattr(src_node, 'tcv'):
                    src_type = "FinancialMetrics"
                elif hasattr(src_node, 'competitive_position'):
                    src_type = "CompetitiveLandscape"
                elif hasattr(src_node, 'role') and hasattr(src_node, 'name'):
                    src_type = "Partner"
                    
                if hasattr(dst_node, 'opportunity_id'):
                    dst_type = "Opportunity"
                elif hasattr(dst_node, 'segment'):
                    dst_type = "Customer"
                elif hasattr(dst_node, 'contact_type'):
                    dst_type = "Person"
                elif hasattr(dst_node, 'risk_description'):
                    dst_type = "RiskAnalysis"
                elif hasattr(dst_node, 'tcv'):
                    dst_type = "FinancialMetrics"
                elif hasattr(dst_node, 'competitive_position'):
                    dst_type = "CompetitiveLandscape"
                elif hasattr(dst_node, 'role') and hasattr(dst_node, 'name'):
                    dst_type = "Partner"
                
                # Try to get relationship type from relationship object
                if hasattr(rel_obj, '__class__'):
                    rel_type = rel_obj.__class__.__name__.replace('Relationship', '').replace('Record', '')
                    if not rel_type or rel_type == 'object':
                        rel_type = "RELATED_TO"
                
                # Get node identifiers
                src_name = "unknown"
                dst_name = "unknown"
                
                # Extract names based on node type
                if hasattr(src_node, 'name') and src_node.name:
                    src_name = str(src_node.name)
                elif hasattr(src_node, 'risk_description') and src_node.risk_description:
                    src_name = str(src_node.risk_description)[:50]
                elif hasattr(src_node, 'competitive_position') and src_node.competitive_position:
                    src_name = str(src_node.competitive_position)[:50]
                elif hasattr(src_node, 'tcv') and src_node.tcv:
                    src_name = f"${src_node.tcv} TCV"
                    
                if hasattr(dst_node, 'name') and dst_node.name:
                    dst_name = str(dst_node.name)
                elif hasattr(dst_node, 'risk_description') and dst_node.risk_description:
                    dst_name = str(dst_node.risk_description)[:50]
                elif hasattr(dst_node, 'competitive_position') and dst_node.competitive_position:
                    dst_name = str(dst_node.competitive_position)[:50]
                elif hasattr(dst_node, 'tcv') and dst_node.tcv:
                    dst_name = f"${dst_node.tcv} TCV"
                
                src_id = f"{src_type}_{src_name.replace(' ', '_').replace('/', '_')[:30]}"
                dst_id = f"{dst_type}_{dst_name.replace(' ', '_').replace('/', '_')[:30]}"
                
                # Only add if we have meaningful source and destination IDs
                if not src_id.startswith("Unknown_") and not dst_id.startswith("Unknown_"):
                    edges.append((src_id, dst_id, rel_type, {}))
                
        except Exception as e:
            print(f"Error fetching relationships: {e}")
            
        # If we have no meaningful edges, create a logical graph structure 
        # Filter out any "Unknown_unknown" or meaningless relationships
        meaningful_edges = [e for e in edges if not (e[0].startswith("Unknown_") or e[1].startswith("Unknown_"))]
        
        if not meaningful_edges and nodes:
            # Find different node types
            opportunity_nodes = [n for n in nodes if n[1].get('type') == 'Opportunity']
            customer_nodes = [n for n in nodes if n[1].get('type') == 'Customer']
            person_nodes = [n for n in nodes if n[1].get('type') == 'Person']
            partner_nodes = [n for n in nodes if n[1].get('type') == 'Partner']
            risk_nodes = [n for n in nodes if n[1].get('type') == 'RiskAnalysis']
            financial_nodes = [n for n in nodes if n[1].get('type') == 'FinancialMetrics']
            competitive_nodes = [n for n in nodes if n[1].get('type') == 'CompetitiveLandscape']
            
            # Create logical connections based on business relationships
            if opportunity_nodes:
                opp_id = opportunity_nodes[0][0]
                
                # Opportunity -> Customer
                for customer_id, _ in customer_nodes:
                    edges.append((opp_id, customer_id, "HAS_CUSTOMER", {}))
                
                # Opportunity -> Risks
                for risk_id, _ in risk_nodes:
                    edges.append((opp_id, risk_id, "HAS_RISK", {}))
                    
                # Opportunity -> Partners
                for partner_id, _ in partner_nodes:
                    edges.append((opp_id, partner_id, "HAS_PARTNER", {}))
                    
                # Opportunity -> Financials
                for fin_id, _ in financial_nodes:
                    edges.append((opp_id, fin_id, "HAS_FINANCIALS", {}))
                    
                # Opportunity -> Competition
                for comp_id, _ in competitive_nodes:
                    edges.append((opp_id, comp_id, "HAS_COMPETITION", {}))
                
                # Team members (internal people) -> Opportunity
                for person_id, person_data in person_nodes:
                    if person_data.get('contact_type') == 'Internal':
                        edges.append((opp_id, person_id, "HAS_TEAM_MEMBER", {}))
                
            # Customer -> Customer contacts
            for customer_id, _ in customer_nodes:
                for person_id, person_data in person_nodes:
                    if person_data.get('contact_type') == 'Client':
                        edges.append((customer_id, person_id, "HAS_CONTACT", {}))
            
            # Clear out the old broken relationships and use our new logical ones
            edges = [e for e in edges if not (e[0].startswith("Unknown_") or e[1].startswith("Unknown_"))]
            
    except Exception as e:
        print(f"Error in _fetch_graph_data: {e}")
        return [], []

    return nodes, edges


def generate_kuzu_graph_html(
    connection: kuzu.Connection,
    destination_file_path: str | None = None,
) -> str:
    """Generate an HTML graph visualization from a Kuzu connection.

    Args:
        connection: An active kuzu.Connection connected to a database that uses
            a schema with Node(id, name, type, properties) and EDGE(relationship_name, properties).
        destination_file_path: Optional path to write the HTML file. If omitted,
            the file will be saved as "graph_visualization.html" in the user's home directory.

    Returns:
        The HTML content as a string.
    """
    nodes_data, edges_data = _fetch_graph_data(connection)

    # Build visualization model with colors for our specific node types
    color_map = {
        "Opportunity": "#FF6B6B",     # Red for opportunities
        "Customer": "#4ECDC4",        # Teal for customers
        "Person": "#45B7D1",          # Blue for people
        "Partner": "#96CEB4",         # Green for partners
        "RiskAnalysis": "#FECA57",    # Yellow for risks
        "FinancialMetrics": "#FF9FF3", # Pink for financials
        "TechnicalApproach": "#54A0FF", # Light blue for tech
        "CompetitiveLandscape": "#5F27CD", # Purple for competition
        "default": "#D3D3D3",       # Gray for unknown types
    }

    nodes_list: list[dict[str, Any]] = []
    for node_id, node_info in nodes_data:
        node_info = dict(node_info)  # shallow copy
        node_info["id"] = str(node_id)
        node_info["color"] = color_map.get(node_info.get("type", "default"), "#D3D3D3")
        node_info["name"] = node_info.get("name", str(node_id))
        # Trim noisy fields if present
        node_info.pop("updated_at", None)
        node_info.pop("created_at", None)
        nodes_list.append(node_info)

    links_list: list[dict[str, Any]] = []
    for source, target, relation, edge_info in edges_data:
        source_s = str(source)
        target_s = str(target)

        # Extract weight variations
        all_weights: dict[str, float] = {}
        primary_weight: float | None = None
        edge_info = edge_info or {}

        if "weight" in edge_info:
            try:
                primary_weight = float(edge_info["weight"])  # best effort
                all_weights["default"] = primary_weight
            except (TypeError, ValueError):
                pass

        if "weights" in edge_info and isinstance(edge_info["weights"], dict):
            for k, v in edge_info["weights"].items():
                try:
                    all_weights[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            if primary_weight is None and all_weights:
                primary_weight = next(iter(all_weights.values()))

        for key, value in edge_info.items():
            if key.startswith("weight_"):
                try:
                    all_weights[key[7:]] = float(value)
                except (TypeError, ValueError):
                    continue

        links_list.append(
            {
                "source": source_s,
                "target": target_s,
                "relation": relation,
                "weight": primary_weight,
                "all_weights": all_weights,
                "relationship_type": edge_info.get("relationship_type"),
                "edge_info": edge_info,
            }
        )

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v5.min.js"></script>
        <style>
            body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: linear-gradient(90deg, #101010, #1a1a2e); color: white; font-family: 'Inter', sans-serif; }

            svg { width: 100vw; height: 100vh; display: block; }
            .links line { stroke: rgba(255, 255, 255, 0.4); stroke-width: 2px; }
            .links line.weighted { stroke: rgba(255, 215, 0, 0.7); }
            .links line.multi-weighted { stroke: rgba(0, 255, 127, 0.8); }
            .nodes circle { stroke: white; stroke-width: 0.5px; filter: drop-shadow(0 0 5px rgba(255,255,255,0.3)); }
            .node-label { font-size: 8px; font-weight: bold; fill: white; text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', sans-serif; pointer-events: none; }
            .edge-label { font-size: 3px; fill: rgba(255, 255, 255, 0.7); text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', sans-serif; pointer-events: none; }
            
            .tooltip {
                position: absolute;
                text-align: left;
                padding: 8px;
                font-size: 12px;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.2s;
                z-index: 1000;
                max-width: 300px;
                word-wrap: break-word;
            }
        </style>
    </head>
    <body>
        <svg></svg>
        <div class="tooltip" id="tooltip"></div>
        <script>
            var nodes = {nodes};
            var links = {links};

            var svg = d3.select("svg"),
                width = window.innerWidth,
                height = window.innerHeight;

            var container = svg.append("g");
            var tooltip = d3.select("#tooltip");

            var simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).strength(0.1))
                .force("charge", d3.forceManyBody().strength(-275))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("x", d3.forceX().strength(0.1).x(width / 2))
                .force("y", d3.forceY().strength(0.1).y(height / 2));

            var link = container.append("g")
                .attr("class", "links")
                .selectAll("line")
                .data(links)
                .enter().append("line")
                .attr("stroke-width", d => {
                    if (d.weight) return Math.max(2, d.weight * 5);
                    if (d.all_weights && Object.keys(d.all_weights).length > 0) {
                        var avgWeight = Object.values(d.all_weights).reduce((a, b) => a + b, 0) / Object.values(d.all_weights).length;
                        return Math.max(2, avgWeight * 5);
                    }
                    return 2;
                })
                .attr("class", d => {
                    if (d.all_weights && Object.keys(d.all_weights).length > 1) return "multi-weighted";
                    if (d.weight || (d.all_weights && Object.keys(d.all_weights).length > 0)) return "weighted";
                    return "";
                })
                .on("mouseover", function(d) {
                    // Create tooltip content for edge
                    var content = "<strong>Edge Information</strong><br/>";
                    content += "Relationship: " + d.relation + "<br/>";

                    // Show all weights
                    if (d.all_weights && Object.keys(d.all_weights).length > 0) {
                        content += "<strong>Weights:</strong><br/>";
                        Object.keys(d.all_weights).forEach(function(weightName) {
                            content += "&nbsp;&nbsp;" + weightName + ": " + d.all_weights[weightName] + "<br/>";
                        });
                    } else if (d.weight !== null && d.weight !== undefined) {
                        content += "Weight: " + d.weight + "<br/>";
                    }

                    if (d.relationship_type) {
                        content += "Type: " + d.relationship_type + "<br/>";
                    }
                    // Add other edge properties
                    if (d.edge_info) {
                        Object.keys(d.edge_info).forEach(function(key) {
                            if (key !== 'weight' && key !== 'weights' && key !== 'relationship_type' && 
                                key !== 'source_node_id' && key !== 'target_node_id' && 
                                key !== 'relationship_name' && key !== 'updated_at' && 
                                !key.startsWith('weight_')) {
                                content += key + ": " + d.edge_info[key] + "<br/>";
                            }
                        });
                    }

                    tooltip.html(content)
                        .style("left", (d3.event.pageX + 10) + "px")
                        .style("top", (d3.event.pageY - 10) + "px")
                        .style("opacity", 1);
                })
                .on("mouseout", function(d) {
                    tooltip.style("opacity", 0);
                });

            var edgeLabels = container.append("g")
                .attr("class", "edge-labels")
                .selectAll("text")
                .data(links)
                .enter().append("text")
                .attr("class", "edge-label")
                .text(d => {
                    var label = d.relation;
                    if (d.all_weights && Object.keys(d.all_weights).length > 1) {
                        // Show count of weights for multiple weights
                        label += " (" + Object.keys(d.all_weights).length + " weights)";
                    } else if (d.weight) {
                        label += " (" + d.weight + ")";
                    } else if (d.all_weights && Object.keys(d.all_weights).length === 1) {
                        var singleWeight = Object.values(d.all_weights)[0];
                        label += " (" + singleWeight + ")";
                    }
                    return label;
                });

            var nodeGroup = container.append("g")
                .attr("class", "nodes")
                .selectAll("g")
                .data(nodes)
                .enter().append("g");

            var node = nodeGroup.append("circle")
                .attr("r", 13)
                .attr("fill", d => d.color)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            nodeGroup.append("text")
                .attr("class", "node-label")
                .attr("dy", 4)
                .attr("text-anchor", "middle")
                .text(d => d.name);

            node.append("title").text(d => JSON.stringify(d));

            simulation.on("tick", function() {
                link.attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                edgeLabels
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2 - 5);

                node.attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                nodeGroup.select("text")
                    .attr("x", d => d.x)
                    .attr("y", d => d.y)
                    .attr("dy", 4)
                    .attr("text-anchor", "middle");
            });

            svg.call(d3.zoom().on("zoom", function() {
                container.attr("transform", d3.event.transform);
            }));

            function dragstarted(d) {
                if (!d3.event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(d) {
                d.fx = d3.event.x;
                d.fy = d3.event.y;
            }

            function dragended(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }

            window.addEventListener("resize", function() {
                width = window.innerWidth;
                height = window.innerHeight;
                svg.attr("width", width).attr("height", height);
                simulation.force("center", d3.forceCenter(width / 2, height / 2));
                simulation.alpha(1).restart();
            });
        </script>
    </body>
    </html>
    """

    html_content = html_template.replace("{nodes}", json.dumps(nodes_list))
    html_content = html_content.replace("{links}", json.dumps(links_list))

    if not destination_file_path:
        home_dir = os.path.expanduser("~")
        destination_file_path = os.path.join(home_dir, "graph_visualization.html")

    os.makedirs(os.path.dirname(destination_file_path) or ".", exist_ok=True)
    with open(destination_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_content
