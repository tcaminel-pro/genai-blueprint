#!/usr/bin/env python3
"""Interactive EKG (Enhanced Knowledge Graph) CLI.

A comprehensive Typer-based CLI for managing a single Kuzu knowledge graph
created from BAML-structured data. Provides commands for adding opportunity data,
querying with Cypher, and exporting visualizations.

Features:
    - Add opportunity data to shared knowledge base
    - Execute interactive Cypher queries
    - Display database schema and statistics
    - Export HTML visualizations with clickable links
    - Rich console output with colors and tables

Commands:
    ekg add --key OPPORTUNITY_KEY         Add opportunity data to KB
    ekg delete                           Delete entire KB
    ekg query                            Interactive Cypher query shell
    ekg info                             Display DB info and schema
    ekg export-html                      Export HTML visualization

Usage Examples:
    ```bash
    # Add opportunity data to knowledge base
    uv run test_graph_cli.py ekg add --key cnes-venus-tma

    # Query the knowledge base interactively
    uv run test_graph_cli.py ekg query

    # Display database information and mapping
    uv run test_graph_cli.py ekg info

    # Export HTML visualization
    uv run test_graph_cli.py ekg export-html
    ```
"""

import shutil
import webbrowser
from pathlib import Path
from typing import Annotated

import kuzu
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Initialize Rich console
console = Console()

# Configuration constants
KV_STORE_ID = "file"
EKG_DB_DIR = Path.home() / "kuzu"
EKG_DB_PATH = EKG_DB_DIR / "ekg_database.db"


def get_ekg_db_path() -> Path:
    """Get the EKG database path.

    Returns:
        Path to the shared EKG database file
    """
    return EKG_DB_PATH


def load_opportunity_data(opportunity_key: str):
    """Load opportunity data from the key-value store.

    Args:
        opportunity_key: The opportunity identifier to load

    Returns:
        ReviewedOpportunity instance or None if not found
    """

    from src.demos.ekg.baml_client.types import (
        ReviewedOpportunity,
    )
    from src.utils.pydantic.kv_store import PydanticStore

    try:
        store = PydanticStore(kvstore_id=KV_STORE_ID, model=ReviewedOpportunity)
        opportunity = store.load_object(opportunity_key)
        return opportunity
    except Exception as e:
        console.print(f"[red]Error loading opportunity data: {e}[/red]")
        return None


def create_graph_configuration() -> object:
    """Create the graph schema configuration.

    Returns:
        GraphSchema with all node and relationship configurations
    """
    # Define nodes - just specify the class and key field

    from src.demos.ekg.baml_client.types import (
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
    from src.demos.ekg.graph_schema import GraphNodeConfig, GraphRelationConfig, create_simplified_schema

    nodes = [
        # Root node
        GraphNodeConfig(baml_class=ReviewedOpportunity, key="start_date"),
        # Regular nodes - field paths auto-deduced
        GraphNodeConfig(baml_class=Opportunity, key="name"),
        GraphNodeConfig(baml_class=Customer, key="name"),
        GraphNodeConfig(baml_class=Person, key="name", deduplication_key="name"),  # Handles both contacts and team
        GraphNodeConfig(baml_class=Partner, key="name"),
        GraphNodeConfig(baml_class=RiskAnalysis, key="risk_description"),
        GraphNodeConfig(
            baml_class=TechnicalApproach,
            key="technical_stack",
            key_generator=lambda data, base: data.get("technical_stack")
            or data.get("architecture")
            or f"{base}_default",
        ),
        GraphNodeConfig(
            baml_class=CompetitiveLandscape,
            key="competitive_position",
            key_generator=lambda data, base: data.get("competitive_position") or f"{base}_competitive_position",
        ),
        # Embedded node - financials will be embedded in Opportunity table
        GraphNodeConfig(baml_class=FinancialMetrics, key="tcv", embed_in_parent=True, embed_prefix="financial_"),
    ]

    # Define relationships - just specify from/to classes and relationship name
    # Field paths are automatically deduced from the model structure
    relations = [
        GraphRelationConfig(from_node=ReviewedOpportunity, to_node=Opportunity, name="REVIEWS"),
        GraphRelationConfig(from_node=Opportunity, to_node=Customer, name="HAS_CUSTOMER"),
        GraphRelationConfig(from_node=Customer, to_node=Person, name="HAS_CONTACT"),
        GraphRelationConfig(from_node=ReviewedOpportunity, to_node=Person, name="HAS_TEAM_MEMBER"),
        GraphRelationConfig(from_node=ReviewedOpportunity, to_node=Partner, name="HAS_PARTNER"),
        GraphRelationConfig(from_node=ReviewedOpportunity, to_node=RiskAnalysis, name="HAS_RISK"),
        GraphRelationConfig(from_node=ReviewedOpportunity, to_node=TechnicalApproach, name="HAS_TECH_STACK"),
        GraphRelationConfig(from_node=ReviewedOpportunity, to_node=CompetitiveLandscape, name="HAS_COMPETITION"),
        # Note: No relationship to FinancialMetrics because it's embedded in Opportunity
    ]

    # Create and validate the schema - this will auto-deduce all field paths
    schema = create_simplified_schema(root_model_class=ReviewedOpportunity, nodes=nodes, relations=relations)

    return schema


def get_db_connection() -> tuple[kuzu.Database, kuzu.Connection] | None:
    """Get database connection to the shared EKG database.

    Returns:
        Tuple of (Database, Connection) or None if database doesn't exist
    """
    db_path = get_ekg_db_path()

    if not db_path.exists():
        return None

    try:
        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)
        return db, conn
    except Exception as e:
        console.print(f"[red]Error connecting to database: {e}[/red]")
        return None


## subcommands


def register_ekg_commands(cli_app: typer.Typer) -> None:
    # app = typer.Typer()
    # app.add_typer(cli_app, name="ekg")
    app = cli_app

    @app.command("kg-add")
    def add_opportunity(
        key: Annotated[str, typer.Option("--key", "-k", help="Opportunity key to add to the EKG database")],
    ) -> None:
        """Add opportunity data to the shared EKG database.

        Loads opportunity data from the key-value store and adds it to the
        shared EKG database, creating the database if it doesn't exist.
        """

        from src.demos.ekg.graph_core import create_graph

        console.print(Panel(f"[bold cyan]Adding Opportunity Data: {key}[/bold cyan]"))

        # Check if opportunity data exists
        console.print("📁 Loading opportunity data...")
        opportunity = load_opportunity_data(key)
        if not opportunity:
            console.print(f"[red]❌ No opportunity data found for key: {key}[/red]")
            console.print("[yellow]💡 Use structured extraction commands to create data first[/yellow]")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Loaded opportunity: [bold]{opportunity.opportunity.name}[/bold]")

        # Get database path
        db_path = get_ekg_db_path()

        # Check if database exists
        db_exists = db_path.exists()
        if not db_exists:
            console.print(f"📂 Creating EKG database directory: {db_path.parent}")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            console.print("[green]✓[/green] New EKG database will be created")
        else:
            console.print("[green]✓[/green] Adding to existing EKG database")

        # Create graph configuration
        console.print("⚙️  Creating graph schema...")
        schema = create_graph_configuration()
        console.print(
            f"[green]✓[/green] Schema created with {len(schema.nodes)} node types and {len(schema.relations)} relationships"
        )

        # Initialize or connect to database
        console.print("🔧 Connecting to EKG database...")
        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)

        # Add opportunity data to the knowledge graph
        console.print(f"🚀 Adding opportunity data: {key}...")
        with console.status("[bold green]Processing graph data..."):
            nodes_dict, relationships = create_graph(conn, opportunity, schema)

        # Display addition summary
        total_nodes = sum(len(node_list) for node_list in nodes_dict.values())
        console.print(Panel("[bold green]✅ Opportunity Data Added Successfully![/bold green]"))

        summary_table = Table(title="Addition Summary")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Count", justify="right", style="magenta")

        summary_table.add_row("Opportunity Added", key)
        summary_table.add_row("Nodes Added", str(total_nodes))
        summary_table.add_row("Relationships Added", str(len(relationships)))
        summary_table.add_row("Node Types", str(len([k for k, v in nodes_dict.items() if v])))
        summary_table.add_row("Database Path", str(db_path))

        console.print(summary_table)

        console.print("\n[green]💡 Next steps:[/green]")
        console.print("   • Query: [bold]ekg query[/bold]")
        console.print("   • Info:  [bold]ekg info[/bold]")
        console.print("   • Export: [bold]ekg export-html[/bold]")

    @app.command("kg-delete")
    def delete_ekg() -> None:
        """Delete the entire EKG database.

        Safely removes the shared database directory after confirmation.
        All opportunity data will be lost.
        """
        console.print(Panel("[bold red]Deleting Entire EKG Database[/bold red]"))

        db_path = get_ekg_db_path()

        if not db_path.exists():
            console.print("[yellow]⚠️  No EKG database found[/yellow]")
            console.print(f"[yellow]Expected path: {db_path}[/yellow]")
            raise typer.Exit(0)

        # Show database info before deletion
        console.print(f"📍 Database location: [bold]{db_path}[/bold]")

        try:
            # Try to get some basic stats
            result = get_db_connection()
            if result:
                db, conn = result
                try:
                    tables_result = conn.execute("CALL show_tables() RETURN *")
                    tables_df = tables_result.get_as_df()
                    node_count = len([row for _, row in tables_df.iterrows() if row.get("type") == "NODE"])
                    rel_count = len([row for _, row in tables_df.iterrows() if row.get("type") == "REL"])
                    console.print(f"📊 Contains {node_count} node tables and {rel_count} relationship tables")

                    # Try to get total record counts
                    total_nodes = 0
                    for _, row in tables_df.iterrows():
                        if row.get("type") == "NODE":
                            try:
                                result = conn.execute(f"MATCH (n:{row['name']}) RETURN count(n) as count")
                                count = result.get_as_df().iloc[0]["count"]
                                total_nodes += count
                            except Exception:
                                pass
                    console.print(f"📊 Total nodes in database: {total_nodes}")
                except Exception:
                    console.print("📊 Database exists but couldn't read statistics")
        except Exception:
            pass

        # Confirmation
        if not Confirm.ask("[bold red]Are you sure you want to delete the ENTIRE EKG database?[/bold red]"):
            console.print("[yellow]Operation cancelled[/yellow]")
            raise typer.Exit(0)

        # Final confirmation for safety
        if not Confirm.ask(
            "[bold red]This will delete ALL opportunity data. This action cannot be undone. Continue?[/bold red]"
        ):
            console.print("[yellow]Operation cancelled[/yellow]")
            raise typer.Exit(0)

        # Delete the database
        console.print("🗑️  Deleting EKG database...")
        try:
            if db_path.is_file():
                db_path.unlink()
            elif db_path.is_dir():
                shutil.rmtree(db_path)
            console.print("[green]✅ EKG database deleted successfully[/green]")
            console.print("[green]You can now start fresh with [bold]ekg add --key <opportunity_key>[/bold][/green]")
        except Exception as e:
            console.print(f"[red]❌ Error deleting database: {e}[/red]")
            raise typer.Exit(1)

    @app.command("kg-query")
    def query_ekg(
        query: Annotated[str | None, typer.Option("--query", "-q", help="Cypher query to execute")] = None,
    ) -> None:
        """Execute Cypher queries on the EKG database.

        If no query is provided, starts an interactive query shell.
        """
        console.print(Panel("[bold cyan]Querying EKG Database[/bold cyan]"))

        # Get database connection
        result = get_db_connection()
        if not result:
            console.print("[red]❌ No EKG database found[/red]")
            console.print("[yellow]💡 Add data first: [bold]ekg add --key <opportunity_key>[/bold][/yellow]")
            raise typer.Exit(1)

        db, conn = result
        console.print("[green]✅ Connected to EKG database[/green]")

        def execute_query(cypher_query: str) -> None:
            """Execute a single Cypher query and display results."""
            if not cypher_query.strip():
                return

            try:
                console.print(f"[dim]Executing: {cypher_query}[/dim]")
                result = conn.execute(cypher_query)
                df = result.get_as_df()

                if df.empty:
                    console.print("[yellow]Query returned no results[/yellow]")
                    return

                # Create a Rich table for results
                table = Table(title=f"Query Results ({len(df)} rows)")

                # Add columns
                for col in df.columns:
                    table.add_column(str(col), style="cyan")

                # Add rows (limit to first 20 for readability)
                max_rows = 20
                for i, (_, row) in enumerate(df.iterrows()):
                    if i >= max_rows:
                        table.add_row(*["..." for _ in df.columns])
                        break
                    table.add_row(*[str(val) for val in row])

                console.print(table)

                if len(df) > max_rows:
                    console.print(f"[dim]Showing first {max_rows} of {len(df)} results[/dim]")

            except Exception as e:
                console.print(f"[red]❌ Query error: {e}[/red]")

        # Execute single query if provided
        if query:
            execute_query(query)
            return

        # Interactive query shell
        console.print("\n[bold green]🔍 Interactive Query Shell[/bold green]")
        console.print("[dim]Enter Cypher queries (type 'exit' or 'quit' to leave, 'help' for examples)[/dim]\n")

        # Sample queries
        sample_queries = [
            "MATCH (n) RETURN labels(n)[0] as NodeType, count(n) as Count",
            "MATCH (o:Opportunity) RETURN o.name, o.status LIMIT 5",
            "MATCH (c:Customer)-[:HAS_CONTACT]->(p:Person) RETURN c.name, p.name, p.role LIMIT 5",
            "MATCH (ro:ReviewedOpportunity)-[:HAS_RISK]->(r:RiskAnalysis) RETURN r.risk_description, r.impact_level LIMIT 3",
        ]

        while True:
            try:
                query_input = Prompt.ask("[bold cyan]cypher>[/bold cyan]")

                if query_input.lower() in ("exit", "quit", "q"):
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                elif query_input.lower() == "help":
                    console.print(Panel("Sample Queries", style="green"))
                    for i, sample in enumerate(sample_queries, 1):
                        console.print(f"{i}. [cyan]{sample}[/cyan]")
                    console.print()
                    continue
                elif query_input.isdigit():
                    # Execute sample query by number
                    query_num = int(query_input) - 1
                    if 0 <= query_num < len(sample_queries):
                        execute_query(sample_queries[query_num])
                    else:
                        console.print("[red]Invalid sample query number[/red]")
                    continue

                execute_query(query_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except EOFError:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break

    @app.command("kg-info")
    def show_info() -> None:
        """Display EKG database information, schema, and BAM class mapping.

        Shows comprehensive information about the EKG database including
        node/relationship counts, schema details, and mapping to BAM classes.
        """
        console.print(Panel("[bold cyan]EKG Database Information[/bold cyan]"))

        # Get database connection
        result = get_db_connection()
        if not result:
            console.print("[red]❌ No EKG database found[/red]")
            console.print("[yellow]💡 Add data first: [bold]ekg add --key <opportunity_key>[/bold][/yellow]")
            raise typer.Exit(1)

        db, conn = result
        console.print("[green]✅ Connected to EKG database[/green]\n")

        # Database location info
        db_path = get_ekg_db_path()
        info_table = Table(title="Database Information")
        info_table.add_column("Property", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="green")

        info_table.add_row("Database Path", str(db_path))
        info_table.add_row("Database Type", "Kuzu Graph Database")
        info_table.add_row("Storage", "Persistent File Storage")

        console.print(info_table)
        console.print()

        # Get schema information
        try:
            tables_result = conn.execute("CALL show_tables() RETURN *")
            tables_df = tables_result.get_as_df()

            node_tables = []
            rel_tables = []

            for _, row in tables_df.iterrows():
                if row.get("type") == "NODE":
                    node_tables.append(row["name"])
                elif row.get("type") == "REL":
                    rel_tables.append(row["name"])

            # Schema overview
            schema_table = Table(title="Schema Overview")
            schema_table.add_column("Component", style="cyan", no_wrap=True)
            schema_table.add_column("Count", justify="right", style="magenta")

            schema_table.add_row("Node Tables", str(len(node_tables)))
            schema_table.add_row("Relationship Tables", str(len(rel_tables)))

            console.print(schema_table)
            console.print()

            # Node statistics
            if node_tables:
                node_stats_table = Table(title="Node Counts")
                node_stats_table.add_column("Node Type", style="cyan", no_wrap=True)
                node_stats_table.add_column("Count", justify="right", style="magenta")

                for node_type in sorted(node_tables):
                    try:
                        result = conn.execute(f"MATCH (n:{node_type}) RETURN count(n) as count")
                        count = result.get_as_df().iloc[0]["count"]
                        node_stats_table.add_row(node_type, str(count))
                    except Exception as e:
                        node_stats_table.add_row(node_type, f"[red]Error: {e}[/red]")

                console.print(node_stats_table)
                console.print()

            # Relationship statistics
            if rel_tables:
                rel_stats_table = Table(title="Relationship Counts")
                rel_stats_table.add_column("Relationship Type", style="cyan", no_wrap=True)
                rel_stats_table.add_column("Count", justify="right", style="magenta")

                for rel_type in sorted(rel_tables):
                    try:
                        result = conn.execute(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                        count = result.get_as_df().iloc[0]["count"]
                        rel_stats_table.add_row(rel_type, str(count))
                    except Exception as e:
                        rel_stats_table.add_row(rel_type, f"[red]Error: {e}[/red]")

                console.print(rel_stats_table)
                console.print()

        except Exception as e:
            console.print(f"[red]Error retrieving schema information: {e}[/red]")

        # BAM Class Mapping
        console.print(Panel("[bold cyan]BAM Class Mapping[/bold cyan]"))

        mapping_table = Table(title="Node Type → BAM Class Mapping")
        mapping_table.add_column("Graph Node Type", style="cyan", no_wrap=True)
        mapping_table.add_column("BAM Python Class", style="green")
        mapping_table.add_column("Description", style="yellow")

        # Define the mapping based on our schema configuration
        baml_mapping = {
            "ReviewedOpportunity": ("ReviewedOpportunity", "Root node containing the complete reviewed opportunity"),
            "Opportunity": ("Opportunity", "Core opportunity information with financial metrics embedded"),
            "Customer": ("Customer", "Customer organization details"),
            "Person": ("Person", "Individual contacts and team members"),
            "Partner": ("Partner", "Partner organization information"),
            "RiskAnalysis": ("RiskAnalysis", "Risk assessment and mitigation details"),
            "TechnicalApproach": ("TechnicalApproach", "Technical implementation approach and stack"),
            "CompetitiveLandscape": ("CompetitiveLandscape", "Competitive positioning and analysis"),
        }

        for node_type, (baml_class, description) in baml_mapping.items():
            mapping_table.add_row(node_type, baml_class, description)

        console.print(mapping_table)
        console.print()

        # Relationship mapping
        rel_mapping_table = Table(title="Relationship Type → Semantic Meaning")
        rel_mapping_table.add_column("Relationship", style="cyan", no_wrap=True)
        rel_mapping_table.add_column("From → To", style="green")
        rel_mapping_table.add_column("Meaning", style="yellow")

        relationship_meanings = {
            "REVIEWS": ("ReviewedOpportunity → Opportunity", "Review relationship to core opportunity"),
            "HAS_CUSTOMER": ("Opportunity → Customer", "Opportunity belongs to customer"),
            "HAS_CONTACT": ("Customer → Person", "Customer contact persons"),
            "HAS_TEAM_MEMBER": ("ReviewedOpportunity → Person", "Internal team members"),
            "HAS_PARTNER": ("ReviewedOpportunity → Partner", "Partner organizations involved"),
            "HAS_RISK": ("ReviewedOpportunity → RiskAnalysis", "Identified risks and mitigations"),
            "HAS_TECH_STACK": ("ReviewedOpportunity → TechnicalApproach", "Technical implementation approach"),
            "HAS_COMPETITION": ("ReviewedOpportunity → CompetitiveLandscape", "Competitive analysis"),
        }

        for rel_type, (direction, meaning) in relationship_meanings.items():
            rel_mapping_table.add_row(rel_type, direction, meaning)

        console.print(rel_mapping_table)

        # Quick query suggestions
        console.print("\n[green]💡 Try these queries:[/green]")
        console.print('   • [bold]ekg query --query "MATCH (n) RETURN labels(n)[0], count(n)"[/bold]')
        console.print("   • [bold]ekg query[/bold] (interactive shell)")

    @app.command("kg-export-html")
    def export_html(
        output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory")] = "/tmp",
        open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open in browser")] = True,
    ) -> None:
        """Export EKG graph visualization as HTML and display clickable link.

        Creates an interactive D3.js visualization of the EKG database
        and saves it to the specified output directory.
        """
        from src.demos.ekg.kuzu_graph_html import generate_html_visualization

        console.print(Panel("[bold cyan]Exporting EKG HTML Visualization[/bold cyan]"))

        # Get database connection
        result = get_db_connection()
        if not result:
            console.print("[red]❌ No EKG database found[/red]")
            console.print("[yellow]💡 Add data first: [bold]ekg add --key <opportunity_key>[/bold][/yellow]")
            raise typer.Exit(1)

        db, conn = result
        console.print("[green]✅ Connected to EKG database[/green]")

        # Prepare output path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        html_filename = "ekg_graph_visualization.html"
        html_file_path = output_path / html_filename

        console.print(f"📁 Output location: [bold]{html_file_path}[/bold]")

        # Generate HTML visualization
        console.print("🎨 Generating interactive visualization...")
        try:
            with console.status("[bold green]Creating HTML visualization..."):
                generate_html_visualization(conn, str(html_file_path), title="EKG Database Visualization")

            console.print("[green]✅ HTML visualization created successfully[/green]")

            # Get file size
            file_size = html_file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Display export summary
            export_table = Table(title="Export Summary")
            export_table.add_column("Property", style="cyan", no_wrap=True)
            export_table.add_column("Value", style="green")

            export_table.add_row("File Location", str(html_file_path))
            export_table.add_row("File Size", f"{file_size_mb:.2f} MB")
            export_table.add_row("Format", "Interactive HTML + D3.js")
            export_table.add_row("Features", "Zoomable, draggable, hover tooltips")

            console.print(export_table)

            # Create clickable link panel
            file_url = f"file://{html_file_path.absolute()}"
            console.print(
                Panel(
                    f"[bold green]🌐 Clickable Link:[/bold green]\n\n"
                    f"[link={file_url}]{file_url}[/link]\n\n"
                    f"[dim]Click the link above or copy-paste into your browser[/dim]",
                    title="HTML Visualization Ready",
                    border_style="green",
                )
            )

            # Optionally open in browser
            if open_browser:
                try:
                    console.print("🌐 Opening in default browser...")
                    webbrowser.open(file_url)
                    console.print("[green]✅ Opened in browser[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not open browser automatically: {e}[/yellow]")
                    console.print("[yellow]Please open the file manually using the link above[/yellow]")

        except Exception as e:
            console.print(f"[red]❌ Error generating visualization: {e}[/red]")
            raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
