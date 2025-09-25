#!/usr/bin/env python3
"""Test script for the Knowledge Graph creation from Pydantic models.

This script demonstrates the complete workflow of creating a knowledge graph
from structured Pydantic data using Kuzu as the graph database.
"""

# Add the src directory to Python path for imports

import kuzu
from baml_client.types import (
    CompetitiveLandscape,
    Customer,
    FinancialMetrics,
    Opportunity,
    Partner,
    Person,
    ReviewedOpportunity,
    RiskAnalysis,
    TechnicalApproach,
)
from graph_core import NodeInfo, RelationInfo, create_graph, restart_database
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.utils.pydantic.kv_store import PydanticStore
from kuzu_graph_html import generate_kuzu_graph_html

console = Console()

# Configuration constants
KV_STORE_ID = "file"
OPPORTUNITY_KEY = "cnes-venus-tma"


def create_configuration() -> tuple[list[NodeInfo], list[RelationInfo]]:
    """Create the node and relationship configuration for the opportunity graph.

    Returns:
        Tuple of (nodes, relations) configurations
    """
    nodes = [
        NodeInfo(baml_class=Opportunity, key="name", indexed=True, field_path="opportunity"),
        NodeInfo(baml_class=Customer, key="name", indexed=True, field_path="opportunity.customer"),
        NodeInfo(
            baml_class=Person,
            key="name",
            indexed=True,
            field_path="opportunity.customer.contacts",
            is_list=True,
            deduplication_key="name",
        ),
        NodeInfo(
            baml_class=Person, key="name", indexed=True, field_path="team", is_list=True, deduplication_key="name"
        ),
        NodeInfo(baml_class=Partner, key="name", indexed=True, field_path="partners", is_list=True),
        NodeInfo(baml_class=RiskAnalysis, key="risk_description", indexed=False, field_path="risks", is_list=True),
        NodeInfo(
            baml_class=FinancialMetrics,
            key="tcv",
            indexed=False,
            field_path="financials",
            key_generator=lambda data, base: str(data.get("tcv", 0.0)),
        ),
        NodeInfo(
            baml_class=TechnicalApproach,
            key="technical_stack",
            indexed=False,
            field_path="tech_stack",
            key_generator=lambda data, base: data.get("technical_stack") or data.get("architecture") or f"{base}_default",
        ),
        NodeInfo(
            baml_class=CompetitiveLandscape,
            key="competitive_position",
            indexed=False,
            field_path="competition",
            key_generator=lambda data, base: data.get("competitive_position") or f"{base}_competitive_position",
        ),
    ]

    relations = [
        RelationInfo(
            from_node=Opportunity,
            to_node=Customer,
            name="HAS_CUSTOMER",
            from_field_path="opportunity",
            to_field_path="opportunity.customer",
        ),
        RelationInfo(
            from_node=Customer,
            to_node=Person,
            name="HAS_CONTACT",
            from_field_path="opportunity.customer",
            to_field_path="opportunity.customer.contacts",
        ),
        RelationInfo(
            from_node=Opportunity,
            to_node=Person,
            name="HAS_TEAM_MEMBER",
            from_field_path="opportunity",
            to_field_path="team",
        ),
        RelationInfo(
            from_node=Opportunity,
            to_node=Partner,
            name="HAS_PARTNER",
            from_field_path="opportunity",
            to_field_path="partners",
        ),
        RelationInfo(
            from_node=Opportunity,
            to_node=RiskAnalysis,
            name="HAS_RISK",
            from_field_path="opportunity",
            to_field_path="risks",
        ),
        RelationInfo(
            from_node=Opportunity,
            to_node=FinancialMetrics,
            name="HAS_FINANCIALS",
            from_field_path="opportunity",
            to_field_path="financials",
        ),
        RelationInfo(
            from_node=Opportunity,
            to_node=TechnicalApproach,
            name="HAS_TECH_STACK",
            from_field_path="opportunity",
            to_field_path="tech_stack",
        ),
        RelationInfo(
            from_node=Opportunity,
            to_node=CompetitiveLandscape,
            name="HAS_COMPETITION",
            from_field_path="opportunity",
            to_field_path="competition",
        ),
    ]

    return nodes, relations


def load_test_data() -> ReviewedOpportunity:
    """Load test opportunity data from the key-value store.

    Returns:
        The loaded ReviewedOpportunity instance
    """
    console.print(Panel("[bold cyan]Loading Test Data[/bold cyan]"))

    store = PydanticStore(kvstore_id=KV_STORE_ID, model=ReviewedOpportunity)
    opportunity = store.load_object(OPPORTUNITY_KEY)

    if not opportunity:
        console.print(f"[red]Error: Could not load opportunity '{OPPORTUNITY_KEY}' from store[/red]")
        exit(1)

    console.print(f"[green]✓[/green] Loaded opportunity: [bold]{opportunity.opportunity.name}[/bold]")
    console.print(f"[green]✓[/green] Status: {opportunity.opportunity.status}")
    console.print(f"[green]✓[/green] Customer: {opportunity.opportunity.customer.name}")
    console.print(f"[green]✓[/green] Team size: {len(opportunity.team)}")
    console.print(f"[green]✓[/green] Partners: {len(opportunity.partners)}")
    console.print(f"[green]✓[/green] Risks: {len(opportunity.risks)}")

    return opportunity


def create_statistics_table(conn: kuzu.Connection, nodes: list[NodeInfo], relations: list[RelationInfo]) -> None:
    """Display comprehensive statistics about the created graph.

    Args:
        conn: The Kuzu database connection
        nodes: Node configurations
        relations: Relation configurations
    """
    console.print(Panel("[bold cyan]Graph Statistics[/bold cyan]"))

    # Node statistics
    node_table = Table(title="Node Counts")
    node_table.add_column("Node Type", style="cyan", no_wrap=True)
    node_table.add_column("Count", justify="right", style="magenta")

    # Get unique node types to avoid duplicates
    unique_node_types = set()
    for node_info in nodes:
        node_type = node_info.baml_class.__name__
        if node_type not in unique_node_types:
            unique_node_types.add(node_type)
            try:
                result = conn.execute(f"MATCH (n:{node_type}) RETURN count(n) as count")
                count = result.get_as_df().iloc[0]["count"]
                node_table.add_row(node_type, str(count))
            except Exception as e:
                node_table.add_row(node_type, f"[red]Error: {e}[/red]")

    console.print(node_table)

    # Relationship statistics
    rel_table = Table(title="Relationship Counts")
    rel_table.add_column("Relationship Type", style="cyan", no_wrap=True)
    rel_table.add_column("Count", justify="right", style="magenta")

    for relation in relations:
        rel_name = relation.name
        try:
            result = conn.execute(f"MATCH ()-[r:{rel_name}]->() RETURN count(r) as count")
            count = result.get_as_df().iloc[0]["count"]
            rel_table.add_row(rel_name, str(count))
        except Exception as e:
            rel_table.add_row(rel_name, f"[red]Error: {e}[/red]")

    console.print(rel_table)


def run_sample_queries(conn: kuzu.Connection) -> None:
    """Run sample queries to demonstrate the graph functionality.

    Args:
        conn: The Kuzu database connection
    """
    console.print(Panel("[bold cyan]Sample Query Results[/bold cyan]"))

    # Opportunity details
    console.print("[bold]Opportunity Details:[/bold]")
    query = "MATCH (o:Opportunity) RETURN o.name as name, o.opportunity_id as id, o.status as status"
    console.print(f"[dim]Query: {query}[/dim]")
    result = conn.execute(query)
    df = result.get_as_df()
    if not df.empty:
        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("ID", style="magenta")
        table.add_column("Status", style="green")
        for _, row in df.iterrows():
            table.add_row(str(row["name"]), str(row["id"]), str(row["status"]))
        console.print(table)
    else:
        console.print("[yellow]No opportunities found[/yellow]")

    # Customer and contacts
    console.print("\n[bold]Customer and Contacts:[/bold]")
    query = """
        MATCH (o:Opportunity)-[:HAS_CUSTOMER]->(c:Customer)-[:HAS_CONTACT]->(p:Person)
        RETURN c.name as customer, p.name as contact, p.role as role
        """
    console.print(f"[dim]Query: {query.strip()}[/dim]")
    result = conn.execute(query)
    df = result.get_as_df()
    if not df.empty:
        table = Table()
        table.add_column("Customer", style="cyan")
        table.add_column("Contact", style="magenta")
        table.add_column("Role", style="green")
        for _, row in df.iterrows():
            table.add_row(str(row["customer"]), str(row["contact"]), str(row["role"]))
        console.print(table)
    else:
        console.print("[yellow]No customer contacts found[/yellow]")

    # Team members
    console.print("\n[bold]Team Members (Top 5):[/bold]")
    query = """
        MATCH (o:Opportunity)-[:HAS_TEAM_MEMBER]->(p:Person)
        RETURN p.name as name, p.role as role, p.organization as org
        LIMIT 5
        """
    console.print(f"[dim]Query: {query.strip()}[/dim]")
    result = conn.execute(query)
    df = result.get_as_df()
    if not df.empty:
        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Organization", style="green")
        for _, row in df.iterrows():
            table.add_row(str(row["name"]), str(row["role"]), str(row["org"]))
        console.print(table)
    else:
        console.print("[yellow]No team members found[/yellow]")

    # Risks
    console.print("\n[bold]Top 3 Risks:[/bold]")
    query = """
        MATCH (o:Opportunity)-[:HAS_RISK]->(r:RiskAnalysis)
        RETURN r.risk_description as description, r.impact_level as impact, r.status as status
        LIMIT 3
        """
    console.print(f"[dim]Query: {query.strip()}[/dim]")
    result = conn.execute(query)
    df = result.get_as_df()
    if not df.empty:
        table = Table()
        table.add_column("Description", style="cyan", max_width=50)
        table.add_column("Impact", style="magenta")
        table.add_column("Status", style="green")
        for _, row in df.iterrows():
            description = str(row["description"])
            if len(description) > 50:
                description = description[:47] + "..."
            table.add_row(description, str(row["impact"]), str(row["status"]))
        console.print(table)
    else:
        console.print("[yellow]No risks found[/yellow]")


def run_advanced_graph_queries(conn: kuzu.Connection) -> None:
    """Run advanced graph queries to showcase graph capabilities.

    Args:
        conn: The Kuzu database connection
    """
    console.print(Panel("[bold cyan]Advanced Graph Analysis[/bold cyan]"))

    # Path analysis - connections between customer contacts and team members
    console.print("[bold]Connectivity Analysis:[/bold]")
    query = """
        MATCH path = (customer:Customer)-[:HAS_CONTACT]->(contact:Person),
                     (opportunity:Opportunity)-[:HAS_TEAM_MEMBER]->(team:Person)
        WHERE (opportunity)-[:HAS_CUSTOMER]->(customer)
        RETURN contact.name as contact_person, team.name as team_member,
               contact.role as contact_role, team.role as team_role
        LIMIT 5
        """
    console.print(f"[dim]Query: {query.strip()}[/dim]")
    result = conn.execute(query)
    df = result.get_as_df()
    if not df.empty:
        table = Table(title="Customer-Team Connections")
        table.add_column("Customer Contact", style="cyan")
        table.add_column("Team Member", style="magenta")
        table.add_column("Contact Role", style="green")
        table.add_column("Team Role", style="yellow")
        for _, row in df.iterrows():
            table.add_row(
                str(row["contact_person"]), str(row["team_member"]), str(row["contact_role"]), str(row["team_role"])
            )
        console.print(table)
    else:
        console.print("[yellow]No connections found[/yellow]")

    # Risk impact analysis
    console.print("\n[bold]Risk Impact Distribution:[/bold]")
    query = """
        MATCH (o:Opportunity)-[:HAS_RISK]->(r:RiskAnalysis)
        RETURN r.impact_level as impact, count(r) as risk_count
        ORDER BY risk_count DESC
        """
    console.print(f"[dim]Query: {query.strip()}[/dim]")
    result = conn.execute(query)
    df = result.get_as_df()
    if not df.empty:
        table = Table(title="Risk Impact Levels")
        table.add_column("Impact Level", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        for _, row in df.iterrows():
            table.add_row(str(row["impact"]), str(row["risk_count"]))
        console.print(table)
    else:
        console.print("[yellow]No risk data found[/yellow]")


def main() -> None:
    """Main test execution function."""
    console.print(Text("🚀 Knowledge Graph Test Suite", style="bold magenta"))
    console.print(Text("─" * 50, style="dim"))

    try:
        # Load test data
        opportunity = load_test_data()

        # Create graph configuration
        nodes, relations = create_configuration()
        console.print(
            f"\n[green]✓[/green] Created configuration with {len(nodes)} node types and {len(relations)} relationship types"
        )

        # Initialize database
        db, conn = restart_database()

        # Create the knowledge graph
        console.print(Panel("[bold cyan]Creating Knowledge Graph[/bold cyan]"))
        nodes_dict, relationships = create_graph(conn, opportunity, nodes, relations)

        # Display statistics
        create_statistics_table(conn, nodes, relations)

        # Run sample queries
        run_sample_queries(conn)

        # Run advanced queries
        run_advanced_graph_queries(conn)

        # Summary
        console.print(Panel("[bold green]Test Completed Successfully![/bold green]"))
        total_nodes = sum(len(node_list) for node_list in nodes_dict.values())
        console.print("[bold]Summary:[/bold]")
        console.print(f"• Total nodes created: {total_nodes}")
        console.print(f"• Total relationships created: {len(relationships)}")
        console.print(f"• Node types: {len([k for k, v in nodes_dict.items() if v])}")
        console.print(f"• Relationship types: {len(set(rel[4] for rel in relationships))}")

        generate_kuzu_graph_html(conn, "ekg_visu.html")

    except Exception as e:
        console.print(Panel(f"[bold red]Test Failed[/bold red]\n\n{str(e)}", style="red"))
        raise


if __name__ == "__main__":
    main()
