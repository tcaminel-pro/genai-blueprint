from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import kuzu
from pydantic import BaseModel
from rich.console import Console

console = Console()


class NodeInfo(BaseModel):
    """Configuration for extracting nodes from a Pydantic model.

    Args:
        baml_class: Pydantic model class this node represents
        key: Primary key field name
        indexed: Whether to index the primary key (no-op currently)
        field_path: Dot path to field in the root model
        is_list: Whether the field is a list
        key_generator: Optional custom key generator when the key is missing
        deduplication_key: Field to use for deduplication
    """

    baml_class: type[BaseModel]
    key: str
    indexed: bool = False
    field_path: str | None = None
    is_list: bool = False
    key_generator: Callable[[Dict[str, Any], str], str] | None = None
    deduplication_key: str | None = None


class RelationInfo(BaseModel):
    """Configuration for extracting relationships from a Pydantic model.

    Args:
        from_node: Source node Pydantic class
        to_node: Target node Pydantic class
        name: Relationship name
        from_field_path: Path to source object in the model
        to_field_path: Path to target object(s) in the model
        from_key_field: Override for the source primary key field
        to_key_field: Override for the target primary key field
        attributes: Optional mapping for relationship attributes (not used yet)
    """

    from_node: type[BaseModel]
    to_node: type[BaseModel]
    name: str
    from_field_path: str | None = None
    to_field_path: str
    from_key_field: str | None = None
    to_key_field: str | None = None
    attributes: dict[str, str] = {}


# Database helpers


def restart_database() -> tuple[kuzu.Database, kuzu.Connection]:
    """Restart the database connection to clear all tables.

    Returns:
        Tuple of (Database, Connection)
    """
    db = kuzu.Database(":memory:")
    conn = kuzu.Connection(db)
    console.print("[yellow]🔄 Database restarted - all tables cleared[/yellow]")
    return db, conn


def create_synthetic_key(data: Dict[str, Any], base_name: str) -> str:
    """Generate a synthetic key when primary key is missing.

    Args:
        data: The node data
        base_name: Node type to prefix the synthetic key

    Returns:
        Generated synthetic key
    """
    return f"{base_name}_{hash(str(sorted(data.items()))) % 10000}"


# Schema


def create_schema(conn: kuzu.Connection, nodes: list[NodeInfo], relations: list[RelationInfo]) -> None:
    """Create node and relationship tables in Kuzu database.

    Creates CREATE NODE TABLE and CREATE REL TABLE statements based on NodeInfo
    and RelationInfo configurations. Handles table drops and recreation for
    idempotency.
    """
    # Drop existing rel tables first
    for relation in relations:
        table_name = relation.name
        try:
            conn.execute(f"DROP TABLE {table_name};")
        except Exception:
            pass

    # Drop node tables
    dropped_tables: set[str] = set()
    for node in nodes:
        table_name = node.baml_class.__name__
        if table_name in dropped_tables:
            continue
        try:
            conn.execute(f"DROP TABLE {table_name};")
            dropped_tables.add(table_name)
        except Exception:
            pass

    # Create node tables
    created_tables: set[str] = set()
    for node in nodes:
        table_name = node.baml_class.__name__
        if table_name in created_tables:
            continue

        key_field = node.key
        fields: list[str] = []
        model_fields = node.baml_class.model_fields

        for field_name, field_info in model_fields.items():
            # Best-effort mapping of Python types to Kuzu types
            ann = field_info.annotation
            if getattr(ann, "__origin__", None) is list:
                fields.append(f"{field_name} STRING[]")
            elif ann in (float,):
                fields.append(f"{field_name} DOUBLE")
            elif ann in (int,):
                fields.append(f"{field_name} INT64")
            else:
                fields.append(f"{field_name} STRING")

        fields_str = ", ".join(fields)
        create_sql = f"CREATE NODE TABLE {table_name}({fields_str}, PRIMARY KEY({key_field}))"
        console.print(f"[cyan]Creating node table:[/cyan] {create_sql}")
        conn.execute(create_sql)
        created_tables.add(table_name)

    # Create relationship tables
    for relation in relations:
        from_table = relation.from_node.__name__
        to_table = relation.to_node.__name__
        rel_name = relation.name
        create_rel_sql = f"CREATE REL TABLE {rel_name}(FROM {from_table} TO {to_table})"
        console.print(f"[cyan]Creating relationship table:[/cyan] {create_rel_sql}")
        conn.execute(create_rel_sql)


# Extraction helpers


def get_field_by_path(obj: Any, path: str) -> Any:
    """Get an attribute by a dot-separated path.

    Args:
        obj: Root object or dict
        path: Dot path like a.b.c

    Returns:
        Value at that path or None if not found
    """
    try:
        current = obj
        for part in path.split("."):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    except (AttributeError, KeyError, TypeError):
        return None


def extract_graph_data(
    model: BaseModel, nodes: list[NodeInfo], relations: list[RelationInfo]
) -> Tuple[Dict[str, List[Dict]], List[Tuple]]:
    """Generic extraction of nodes and relationships from any Pydantic model.

    Args:
        model: Pydantic model instance
        nodes: Node extraction configurations
        relations: Relationship extraction configurations

    Returns:
        nodes_dict: Mapping of node type to list of property dicts
        relationships: Tuples of (from_type, from_key, to_type, to_key, rel_name)
    """
    nodes_dict: Dict[str, List[Dict]] = {}
    relationships: List[Tuple] = []
    node_registry: Dict[str, set[str]] = {}

    # Init buckets
    for node_info in nodes:
        node_type = node_info.baml_class.__name__
        nodes_dict[node_type] = []
        node_registry[node_type] = set()

    # Nodes
    for node_info in nodes:
        node_type = node_info.baml_class.__name__
        field_data = get_field_by_path(model, node_info.field_path) if node_info.field_path else model
        if field_data is None:
            continue
        items = field_data if node_info.is_list else [field_data]

        for item in items:
            if item is None:
                continue

            if hasattr(item, "model_dump"):
                item_data = item.model_dump()
            elif isinstance(item, dict):
                item_data = item.copy()
            else:
                continue

            # Primary key
            if node_info.key_generator:
                key_value = node_info.key_generator(item_data, node_type)
            else:
                key_value = item_data.get(node_info.key)

            if key_value is None or key_value == "":
                key_value = create_synthetic_key(item_data, node_type)
                item_data[node_info.key] = key_value

            # Dedup
            dedup_key = node_info.deduplication_key or node_info.key
            dedup_value = item_data.get(dedup_key)
            if dedup_value and dedup_value not in node_registry[node_type]:
                nodes_dict[node_type].append(item_data)
                node_registry[node_type].add(str(dedup_value))
            elif not dedup_value:
                nodes_dict[node_type].append(item_data)

    # Relationships
    for relation_info in relations:
        from_type = relation_info.from_node.__name__
        to_type = relation_info.to_node.__name__

        from_data = get_field_by_path(model, relation_info.from_field_path) if relation_info.from_field_path else model
        to_data = get_field_by_path(model, relation_info.to_field_path)
        if from_data is None or to_data is None:
            continue

        from_node_info = next((n for n in nodes if n.baml_class.__name__ == from_type), None)
        to_node_info = next((n for n in nodes if n.baml_class.__name__ == to_type), None)
        if not from_node_info or not to_node_info:
            continue

        from_dict = from_data.model_dump() if hasattr(from_data, "model_dump") else from_data
        from_key_field = relation_info.from_key_field or from_node_info.key
        from_key = from_dict.get(from_key_field)
        if from_node_info.key_generator and from_key is None:
            from_key = from_node_info.key_generator(from_dict, from_type)

        to_items = to_data if isinstance(to_data, list) else [to_data]
        for to_item in to_items:
            if to_item is None:
                continue
            to_dict = to_item.model_dump() if hasattr(to_item, "model_dump") else to_item
            to_key_field = relation_info.to_key_field or to_node_info.key
            to_key = to_dict.get(to_key_field)
            if to_node_info.key_generator and to_key is None:
                to_key = to_node_info.key_generator(to_dict, to_type)

            if from_key and to_key:
                relationships.append((from_type, str(from_key), to_type, str(to_key), relation_info.name))

    return nodes_dict, relationships


# Loading


def load_graph_data(
    conn: kuzu.Connection, nodes_dict: Dict[str, List[Dict]], relationships: List[Tuple], nodes: list[NodeInfo]
) -> None:
    """Load nodes and relationships into Kuzu database.

    Uses CREATE statements to insert data.
    """
    # Nodes
    for node_type, node_list in nodes_dict.items():
        if not node_list:
            continue
        console.print(f"[green]Loading {len(node_list)} {node_type} nodes...[/green]")
        for node_data in node_list:
            # Find the node configuration to get the primary key field
            node_info = next((n for n in nodes if n.baml_class.__name__ == node_type), None)
            primary_key_field = node_info.key if node_info else None
            
            cleaned_data: Dict[str, str] = {}
            for key, value in node_data.items():
                if value is None:
                    # Skip NULL primary keys to avoid constraint violations
                    if key == primary_key_field:
                        console.print(f"[yellow]Warning: Skipping {node_type} node with NULL primary key[/yellow]")
                        break
                    cleaned_data[key] = "NULL"
                elif isinstance(value, str):
                    escaped = value.replace("'", "\\'")
                    cleaned_data[key] = f"'{escaped}'"
                elif isinstance(value, list):
                    str_list: list[str] = []
                    for v in value:
                        escaped_v = str(v).replace("'", "\\'")
                        str_list.append(f"'{escaped_v}'")
                    cleaned_data[key] = f"[{','.join(str_list)}]"
                else:
                    cleaned_data[key] = str(value)
            else:
                # This else clause is executed only if the for loop completes without break
                fields = ", ".join([f"{k}: {v}" for k, v in cleaned_data.items()])
                create_sql = f"CREATE (:{node_type} {{{fields}}})"
                try:
                    conn.execute(create_sql)
                except Exception as e:
                    console.print(f"[red]Error creating {node_type} node:[/red] {e}")
                    console.print(f"[dim]SQL: {create_sql}[/dim]")

    # Relationships
    console.print(f"[green]Loading {len(relationships)} relationships...[/green]")
    for from_type, from_key, to_type, to_key, rel_name in relationships:
        from_key_escaped = from_key.replace("'", "\\'")
        to_key_escaped = to_key.replace("'", "\\'")

        from_node_key = next(n.key for n in nodes if n.baml_class.__name__ == from_type)
        to_node_key = next(n.key for n in nodes if n.baml_class.__name__ == to_type)

        match_sql = f"""
        MATCH (from:{from_type}), (to:{to_type})
        WHERE from.{from_node_key} = '{from_key_escaped}'
          AND to.{to_node_key} = '{to_key_escaped}'
        CREATE (from)-[:{rel_name}]->(to)
        """
        try:
            conn.execute(match_sql)
        except Exception as e:
            console.print(f"[red]Error creating {rel_name} relationship:[/red] {e}")
            console.print(f"[dim]SQL: {match_sql}[/dim]")


# Orchestration


def create_graph(
    conn: kuzu.Connection, model: BaseModel, nodes: list[NodeInfo], relations: list[RelationInfo]
) -> tuple[Dict[str, List[Dict]], List[Tuple]]:
    """Create a knowledge graph from a Pydantic model in Kuzu database.

    Args:
        model: Root instance to convert
        nodes: Node type definitions
        relations: Relationship definitions

    Returns:
        nodes_dict and relationships that were used to populate the graph
    """
    console.print("[bold]Creating database schema...[/bold]")
    create_schema(conn, nodes, relations)

    console.print("[bold]Extracting graph data...[/bold]")
    nodes_dict, relationships = extract_graph_data(model, nodes, relations)

    console.print("[bold]Loading data into graph...[/bold]")
    load_graph_data(conn, nodes_dict, relationships, nodes)

    total_nodes = sum(len(node_list) for node_list in nodes_dict.values())
    console.print("\n[bold green]Graph creation complete![/bold green]")
    console.print(f"[green]Total nodes:[/green] {total_nodes}")
    console.print(f"[green]Total relationships:[/green] {len(relationships)}")
    return nodes_dict, relationships

