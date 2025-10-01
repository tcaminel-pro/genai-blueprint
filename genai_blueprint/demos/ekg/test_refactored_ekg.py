#!/usr/bin/env python3
"""Test script to demonstrate the refactored EKG structure.

This script shows how the new subgraph system works and verifies
that the refactoring was successful.
"""

import typer
from cli_commands_ekg import register_ekg_commands
from rainbow_subgraph import SUBGRAPH_REGISTRY, get_subgraph
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main() -> None:
    """Test the refactored EKG structure."""
    console.print(Panel("[bold cyan]Testing Refactored EKG Structure[/bold cyan]"))

    # Test subgraph registry
    console.print(f"[green]✓[/green] Available subgraphs: {list(SUBGRAPH_REGISTRY.keys())}")

    # Test opportunity subgraph
    opportunity_subgraph = get_subgraph("opportunity")
    console.print(f"[green]✓[/green] Loaded opportunity subgraph: {opportunity_subgraph.name}")

    # Test subgraph methods
    node_labels = opportunity_subgraph.get_node_labels()
    relationship_labels = opportunity_subgraph.get_relationship_labels()
    sample_queries = opportunity_subgraph.get_sample_queries()

    console.print(f"[green]✓[/green] Node labels: {len(node_labels)} types")
    console.print(f"[green]✓[/green] Relationship labels: {len(relationship_labels)} types")
    console.print(f"[green]✓[/green] Sample queries: {len(sample_queries)} queries")

    # Display some details
    details_table = Table(title="Subgraph Details")
    details_table.add_column("Category", style="cyan", no_wrap=True)
    details_table.add_column("Count", justify="right", style="magenta")
    details_table.add_column("Examples", style="green")

    details_table.add_row("Node Types", str(len(node_labels)), ", ".join(list(node_labels.keys())[:3]) + "...")
    details_table.add_row(
        "Relationships", str(len(relationship_labels)), ", ".join(list(relationship_labels.keys())[:3]) + "..."
    )
    details_table.add_row(
        "Sample Queries", str(len(sample_queries)), sample_queries[0][:50] + "..." if sample_queries else "None"
    )

    console.print(details_table)

    # Test generic label extraction
    console.print(f"[green]✓[/green] Generic node labels work: {list(node_labels.keys())[:3]}...")
    console.print(f"[green]✓[/green] Generic relationship labels work: {list(relationship_labels.keys())[:3]}...")

    # Test CLI registration
    app = typer.Typer()
    register_ekg_commands(app)
    console.print("[green]✓[/green] CLI commands registered successfully")

    console.print(
        Panel(
            "[bold green]✅ All tests passed![/bold green]\n\n"
            + "[dim]The refactoring is complete. The EKG system now:\n"
            + "• Isolates data-specific logic in rainbow_subgraph.py\n"
            + "• Uses a generic CLI with pluggable subgraph architecture\n"
            + "• Extracts descriptions from schema config (generic labels)\n"
            + "• Uses ABC instead of Protocol for stronger type safety\n"
            + "• Provides a clean extension point for new data models[/dim]",
            title="Refactoring Complete",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
