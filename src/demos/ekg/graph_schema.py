"""Simplified graph schema configuration API.

This module provides a refactored approach to defining graph schemas with minimal
configuration required from users. It automatically introspects Pydantic models
to derive field paths and relationships, reducing boilerplate and errors.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel, model_validator


class GraphNodeConfig(BaseModel):
    """Simplified node configuration for graph creation.

    Only requires the essential information that cannot be auto-deduced:
    - Which Pydantic class to create nodes for
    - Which field to use as primary key
    - Optional customizations like key generation and embedding

    All field paths, excluded fields, and list detection are automatically
    determined by introspecting the Pydantic model structure.
    """

    baml_class: Type[BaseModel]
    key: str
    description: str = ""
    embed_in_parent: bool = False
    embed_prefix: str = ""
    key_generator: Optional[Callable[[Dict[str, Any], str], str]] = None
    deduplication_key: Optional[str] = None

    # Auto-deduced attributes (populated during schema validation)
    field_paths: List[str] = []  # All paths where this class appears in the root model
    is_list_at_paths: Dict[str, bool] = {}  # Whether it's a list at each path
    excluded_fields: Set[str] = set()  # Auto-computed based on relationships

    def model_post_init(self, __context: Any) -> None:
        """Initialize auto-deduced fields after model creation."""
        if self.embed_in_parent and not self.embed_prefix:
            self.embed_prefix = f"{self.baml_class.__name__.lower()}_"


class GraphRelationConfig(BaseModel):
    """Simplified relationship configuration.

    Only requires the essential relationship information:
    - Source and target node classes
    - Relationship name

    All field paths are automatically deduced from the Pydantic model structure.
    """

    from_node: Type[BaseModel]
    to_node: Type[BaseModel]
    name: str
    description: str = ""

    # Auto-deduced attributes (populated during schema validation)
    field_paths: List[Tuple[str, str]] = []  # (from_path, to_path) pairs


class GraphSchema(BaseModel):
    """Complete graph schema with validation and auto-deduction capabilities."""

    root_model_class: Type[BaseModel]
    nodes: List[GraphNodeConfig]
    relations: List[GraphRelationConfig]

    # Validation results
    _model_field_map: Dict[Type[BaseModel], Dict[str, Any]] = {}
    _warnings: List[str] = []

    @model_validator(mode="after")
    def validate_and_deduce_schema(self) -> "GraphSchema":
        """Validate schema coherence and auto-deduce missing information."""
        self._build_model_field_map()
        self._deduce_node_field_paths()
        self._deduce_relation_field_paths()
        self._compute_excluded_fields()
        self._validate_coherence()
        return self

    def _build_model_field_map(self) -> None:
        """Build a map of all reachable Pydantic model classes and their fields."""
        visited = set()

        def explore_model(model_class: Type[BaseModel], path: str = ""):
            if model_class in visited:
                return
            visited.add(model_class)

            if not hasattr(model_class, "model_fields"):
                return

            self._model_field_map[model_class] = {}

            for field_name, field_info in model_class.model_fields.items():
                field_path = f"{path}.{field_name}" if path else field_name
                annotation = field_info.annotation

                # Handle List[Model] annotations
                if get_origin(annotation) is list:
                    args = get_args(annotation)
                    inner_type = args[0] if args else None

                    # Handle ForwardRef by trying to resolve it to a real class
                    if inner_type is not None and hasattr(inner_type, "__forward_arg__"):
                        # Try to find the class by name in the model's module
                        try:
                            forward_name = inner_type.__forward_arg__
                            if hasattr(model_class.__module__, "__dict__"):
                                import sys

                                module = sys.modules.get(model_class.__module__)
                                if module and hasattr(module, forward_name):
                                    inner_type = getattr(module, forward_name)
                        except (AttributeError, KeyError):
                            pass

                    if inner_type is not None and hasattr(inner_type, "model_fields"):
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": inner_type,
                            "is_list": True,
                            "annotation": annotation,
                        }
                        explore_model(inner_type, field_path)
                    else:
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": annotation,
                            "is_list": True,
                            "annotation": annotation,
                        }
                # Handle Optional[Model] and Union[Model, None]
                elif get_origin(annotation) is Union:
                    args = get_args(annotation)
                    non_none_args = [arg for arg in args if arg is not type(None)]
                    # Unwrap Optional[List[T]] or Union[List[T], None]
                    if len(non_none_args) == 1 and get_origin(non_none_args[0]) is list:
                        inner_args = get_args(non_none_args[0])
                        inner = inner_args[0] if inner_args else None

                        # Handle ForwardRef in Optional[List[ForwardRef]]
                        if inner is not None and hasattr(inner, "__forward_arg__"):
                            try:
                                forward_name = inner.__forward_arg__
                                import sys

                                module = sys.modules.get(model_class.__module__)
                                if module and hasattr(module, forward_name):
                                    inner = getattr(module, forward_name)
                            except (AttributeError, KeyError):
                                pass

                        if inner is not None and hasattr(inner, "model_fields"):
                            self._model_field_map[model_class][field_name] = {
                                "path": field_path,
                                "type": inner,
                                "is_list": True,
                                "annotation": annotation,
                            }
                            explore_model(inner, field_path)
                            continue

                    # Unwrap Optional[T]
                    if len(non_none_args) == 1 and hasattr(non_none_args[0], "model_fields"):
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": non_none_args[0],
                            "is_list": False,
                            "annotation": annotation,
                        }
                        explore_model(non_none_args[0], field_path)
                    else:
                        self._model_field_map[model_class][field_name] = {
                            "path": field_path,
                            "type": annotation,
                            "is_list": False,
                            "annotation": annotation,
                        }
                # Handle direct Model references
                elif hasattr(annotation, "model_fields"):
                    self._model_field_map[model_class][field_name] = {
                        "path": field_path,
                        "type": annotation,
                        "is_list": False,
                        "annotation": annotation,
                    }
                    explore_model(annotation, field_path)
                else:
                    # Primitive field
                    self._model_field_map[model_class][field_name] = {
                        "path": field_path,
                        "type": annotation,
                        "is_list": False,
                        "annotation": annotation,
                    }

        explore_model(self.root_model_class)

    def _deduce_node_field_paths(self) -> None:
        """Auto-deduce field paths for all node configurations."""
        for node_config in self.nodes:
            node_config.field_paths = []
            node_config.is_list_at_paths = {}

            # Special case: root model
            if node_config.baml_class == self.root_model_class:
                node_config.field_paths = [""]  # Empty path = root
                node_config.is_list_at_paths[""] = False
                continue

            # Find all paths where this class appears
            for model_class, fields in self._model_field_map.items():
                for field_name, field_info in fields.items():
                    if field_info["type"] == node_config.baml_class:
                        path = field_info["path"]
                        is_list = field_info["is_list"]

                        node_config.field_paths.append(path)
                        node_config.is_list_at_paths[path] = is_list

    def _deduce_relation_field_paths(self) -> None:
        """Auto-deduce field paths for all relationship configurations."""
        for relation_config in self.relations:
            relation_config.field_paths = []

            # Find all possible paths between from_node and to_node
            from_node_paths = self._get_node_paths(relation_config.from_node)
            to_node_paths = self._get_node_paths(relation_config.to_node)

            # Find logical connections
            for from_path in from_node_paths:
                for to_path in to_node_paths:
                    if self._is_valid_relationship_path(from_path, to_path, relation_config):
                        relation_config.field_paths.append((from_path, to_path))

    def _get_node_paths(self, node_class: Type[BaseModel]) -> List[str]:
        """Get all field paths for a given node class."""
        node_config = next((n for n in self.nodes if n.baml_class == node_class), None)
        return node_config.field_paths if node_config else []

    def _is_valid_relationship_path(self, from_path: str, to_path: str, relation_config: GraphRelationConfig) -> bool:
        """Check if a relationship path makes logical sense."""
        # Root to anything is valid
        if from_path == "":
            return True

        # Check if to_path is a sub-path of from_path or vice versa
        if to_path.startswith(from_path + ".") or from_path.startswith(to_path + "."):
            return True

        # Check if they share a common parent path
        from_parts = from_path.split(".")
        to_parts = to_path.split(".")

        # Find common prefix
        common_len = 0
        for i in range(min(len(from_parts), len(to_parts))):
            if from_parts[i] == to_parts[i]:
                common_len = i + 1
            else:
                break

        # They're related if they have at least one common parent
        return common_len > 0

    def _compute_excluded_fields(self) -> None:
        """Compute which fields should be excluded from each node based on relationships."""
        for node_config in self.nodes:
            excluded_fields = set()

            # Find all fields that are handled by relationships
            for relation_config in self.relations:
                if relation_config.from_node == node_config.baml_class:
                    # Fields that point to other nodes should be excluded
                    for from_path, to_path in relation_config.field_paths:
                        # Extract the field name from the path
                        if to_path and "." in to_path:
                            if from_path == "":
                                # Root node excluding direct field
                                field_name = to_path.split(".")[0]
                                excluded_fields.add(field_name)
                            else:
                                # Get relative field name
                                if to_path.startswith(from_path + "."):
                                    relative_path = to_path[len(from_path) + 1 :]
                                    field_name = relative_path.split(".")[0]
                                    excluded_fields.add(field_name)
                        elif to_path and "." not in to_path:
                            # Direct field reference
                            if from_path == "":
                                excluded_fields.add(to_path)

            # Also exclude embedded fields
            for other_node in self.nodes:
                if other_node.embed_in_parent and other_node.baml_class != node_config.baml_class:
                    # Find if this other node is embedded in our node
                    other_paths = other_node.field_paths
                    our_paths = node_config.field_paths

                    for other_path in other_paths:
                        for our_path in our_paths:
                            if other_path.startswith(our_path + ".") or (our_path == "" and "." in other_path):
                                # The other node is nested under our node
                                if our_path == "":
                                    field_name = other_path.split(".")[0]
                                else:
                                    relative = other_path[len(our_path) + 1 :] if our_path else other_path
                                    field_name = relative.split(".")[0]
                                excluded_fields.add(field_name)

            node_config.excluded_fields = excluded_fields

    def _validate_coherence(self) -> None:
        """Validate that the schema configuration is coherent with the Pydantic model."""
        warnings_list = []

        # Check that all referenced classes in relationships have node configurations
        referenced_classes = set()
        for relation in self.relations:
            referenced_classes.add(relation.from_node)
            referenced_classes.add(relation.to_node)

        configured_classes = {node.baml_class for node in self.nodes}
        missing_classes = referenced_classes - configured_classes

        if missing_classes:
            for cls in missing_classes:
                warnings_list.append(f"Class {cls.__name__} is referenced in relationships but has no GraphNodeConfig")

        # Check for duplicate relationships between the same classes
        relation_pairs = {}
        for relation in self.relations:
            key = (relation.from_node, relation.to_node)
            if key in relation_pairs:
                relation_pairs[key].append(relation.name)
            else:
                relation_pairs[key] = [relation.name]

        for (from_cls, to_cls), names in relation_pairs.items():
            if len(names) > 1:
                warnings_list.append(
                    f"Multiple relationships defined between {from_cls.__name__} and {to_cls.__name__}: {', '.join(names)}"
                )

        # Check that field paths were found for all nodes
        for node in self.nodes:
            if not node.field_paths and node.baml_class != self.root_model_class:
                warnings_list.append(f"No field paths found for {node.baml_class.__name__} in the model structure")

        # Check that field paths were found for relationships
        for relation in self.relations:
            if not relation.field_paths:
                warnings_list.append(
                    f"No valid field paths found for relationship {relation.name} "
                    f"between {relation.from_node.__name__} and {relation.to_node.__name__}"
                )

        # Store warnings
        self._warnings = warnings_list

        # Emit warnings
        for warning_msg in warnings_list:
            warnings.warn(f"Graph schema validation: {warning_msg}", UserWarning)

    def get_warnings(self) -> List[str]:
        """Get all validation warnings."""
        return self._warnings.copy()

    def print_schema_summary(self) -> None:
        """Print a summary of the deduced schema configuration."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        console.print(Panel(f"[bold cyan]Graph Schema Summary for {self.root_model_class.__name__}[/bold cyan]"))

        # Nodes table
        nodes_table = Table(title="Node Configurations")
        nodes_table.add_column("Class", style="cyan")
        nodes_table.add_column("Key Field", style="magenta")
        nodes_table.add_column("Field Paths", style="green")
        nodes_table.add_column("Excluded Fields", style="yellow")

        for node in self.nodes:
            paths_str = ", ".join(node.field_paths) if node.field_paths else "ROOT"
            excluded_str = ", ".join(sorted(node.excluded_fields)) if node.excluded_fields else "None"
            nodes_table.add_row(node.baml_class.__name__, node.key, paths_str, excluded_str)

        console.print(nodes_table)

        # Relations table
        relations_table = Table(title="Relationship Configurations")
        relations_table.add_column("Name", style="cyan")
        relations_table.add_column("From → To", style="magenta")
        relations_table.add_column("Field Path Pairs", style="green")

        for relation in self.relations:
            from_to = f"{relation.from_node.__name__} → {relation.to_node.__name__}"
            paths_str = (
                "; ".join([f"{fp} → {tp}" for fp, tp in relation.field_paths]) if relation.field_paths else "None"
            )
            relations_table.add_row(relation.name, from_to, paths_str)

        console.print(relations_table)

        # Warnings
        if self._warnings:
            console.print("\n[bold red]Warnings:[/bold red]")
            for warning in self._warnings:
                console.print(f"⚠️  {warning}")


def create_simplified_schema(
    root_model_class: Type[BaseModel], nodes: List[GraphNodeConfig], relations: List[GraphRelationConfig]
) -> GraphSchema:
    """Create and validate a simplified graph schema.

    Args:
        root_model_class: The root Pydantic model class
        nodes: List of node configurations
        relations: List of relationship configurations

    Returns:
        Validated GraphSchema with auto-deduced field paths and excluded fields
    """
    return GraphSchema(root_model_class=root_model_class, nodes=nodes, relations=relations)
