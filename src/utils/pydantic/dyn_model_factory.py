"""Create Pydantic classes dynamically from YAML definitions.

This module provides functionality to dynamically create Pydantic model classes
from YAML configuration files. It handles type conversion, nested models,
and field validation rules.
"""

from typing import Any, Dict, List, Type, Union

import yaml
from pydantic import BaseModel, Field, create_model
from rich import print


class PydanticModelFactory:
    """Factory for creating Pydantic classes from YAML definitions.

    Converts YAML schema definitions into dynamically generated Pydantic model
    classes. Supports nested models, list types, and field validation configuration.

    Example:
        ```yaml
        Person:
          description: "A person model"
          fields:
            name:
              type: str
              required: true
            age:
              type: int
              required: false
        ```

        ```python
        factory = PydanticModelFactory()
        Person = factory.create_class_from_dict(yaml_data, "Person")
        person = Person(name="John", age=30)
        ```
    """

    def __init__(self) -> None:
        self.created_classes: Dict[str, Type[BaseModel] | None] = {}

    def yaml_type_to_python_type(self, yaml_type: str) -> Any:
        """Convert YAML type string to Python type.

        Supports basic types (str, int, bool, etc.) and generic types (List[T], Dict[K,V]).

        Args:
            yaml_type: The type string from YAML (e.g., "str", "list[int]", "dict[str, int]")

        Returns:
            The corresponding Python type object.

        Examples:
            >>> factory.yaml_type_to_python_type("str")
            <class 'str'>
            >>> factory.yaml_type_to_python_type("list[int]")
            typing.List[int]
        """
        from typing import Dict, List

        type_mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "bool": bool,
            "boolean": bool,
            "list": List,
            "dict": Dict,
            "List": List,
            "Dict": Dict,
        }

        if yaml_type.lower().startswith(("list[", "dict[")):
            if yaml_type.lower().startswith("list["):
                inner_type = yaml_type[5:-1]
                return List[self.yaml_type_to_python_type(inner_type)]
            else:  # dict[
                inner_types = yaml_type[5:-1].split(",")
                key_type = self.yaml_type_to_python_type(inner_types[0].strip())
                value_type = self.yaml_type_to_python_type(inner_types[1].strip())
                return Dict[key_type, value_type]

        return type_mapping.get(yaml_type, str)

    def create_class_from_dict(self, yaml_data: dict, class_name: str) -> Type[BaseModel]:
        """Create a Pydantic class from dictionary data.

        Processes YAML schema definitions and creates corresponding Pydantic model classes.
        Handles circular dependencies by creating placeholder classes first.

        Args:
            yaml_data: Dictionary containing class definitions with field specifications.
            class_name: Name of the specific class to return from the definitions.

        Returns:
            A dynamically created Pydantic model class.

        Raises:
            ValueError: If the specified class cannot be created.

        Example:
            ```python
            yaml_data = {
                "Person": {
                    "description": "A person model",
                    "fields": {
                        "name": {"type": "str", "required": True},
                        "age": {"type": "int", "required": False}
                    }
                }
            }
            person_class = factory.create_class_from_dict(yaml_data, "Person")
            ```
        """
        self.created_classes.clear()

        # First pass: create all class definitions
        for cls_name, _ in yaml_data.items():
            if cls_name not in self.created_classes:
                # Create placeholder classes to handle circular dependencies
                self.created_classes[cls_name] = None

        # Second pass: create actual classes
        for cls_name, cls_def in yaml_data.items():
            if cls_name not in self.created_classes or self.created_classes[cls_name] is None:
                self._create_class(cls_name, cls_def, yaml_data)

        result = self.created_classes[class_name]
        if result is None:
            raise ValueError(f"Failed to create class '{class_name}'")
        return result

    def _create_model(self, model_name: str, description: str, fields: dict) -> Type[BaseModel]:
        """Create a Pydantic model with programmatic configuration.

        Args:
            model_name: Name of the model class.
            description: Documentation string for the class.
            fields: Dictionary of field definitions (name: (type, default)).

        Returns:
            The created Pydantic model class.
        """
        from pydantic import ConfigDict

        config_dict = ConfigDict(extra="allow")

        if description:
            model_class = create_model(model_name, __config__=config_dict, __base__=BaseModel, **fields)
            model_class.__doc__ = description
            return model_class
        return create_model(model_name, __config__=config_dict, __base__=BaseModel, **fields)

    def _create_class(self, class_name: str, class_def: Dict[str, Any], yaml_data: dict) -> Type[BaseModel]:
        """Create a single Pydantic class from definition.

        Processes field definitions, handles nested models, and creates the final Pydantic class.

        Args:
            class_name: Name of the class to create.
            class_def: Dictionary containing class definition and fields.
            yaml_data: Complete YAML data for resolving nested class references.

        Returns:
            The created Pydantic model class.

        Raises:
            ValueError: If the class definition is invalid or cannot be processed.
        """
        if class_name in self.created_classes and self.created_classes[class_name] is not None:
            return self.created_classes[class_name]

        # Handle case where class_def is a string (shouldn't happen with proper YAML)
        if isinstance(class_def, str):
            raise ValueError(
                f"Invalid class definition for '{class_name}'. Expected dict with 'fields' key, got string: {class_def}"
            )

        description = class_def.get("description", "")
        fields_def = class_def.get("fields", class_def)

        # Ensure fields_def is a dict
        if not isinstance(fields_def, dict):
            raise ValueError(
                f"Invalid fields definition for '{class_name}'. Expected dict, got {type(fields_def)}: {fields_def}"
            )

        fields = {}
        for field_name, field_def in fields_def.items():
            if isinstance(field_def, dict):
                yaml_type = field_def.get("type", "str")
                is_required = field_def.get("required", False)

                if yaml_type.lower().startswith("list["):
                    inner_type = yaml_type[5:-1]
                    if inner_type in yaml_data:
                        if inner_type not in self.created_classes or self.created_classes[inner_type] is None:
                            self._create_class(inner_type, yaml_data[inner_type], yaml_data)
                        field_type = List[self.created_classes[inner_type]]
                    else:
                        field_type = self.yaml_type_to_python_type(yaml_type)
                elif yaml_type in yaml_data:
                    nested_class_name = yaml_type
                    if nested_class_name not in self.created_classes or self.created_classes[nested_class_name] is None:
                        self._create_class(nested_class_name, yaml_data[nested_class_name], yaml_data)
                    field_type = self.created_classes[nested_class_name]
                else:
                    field_type = self.yaml_type_to_python_type(yaml_type)

                field_info = {}
                if "description" in field_def:
                    field_info["description"] = field_def["description"]

                if is_required:
                    fields[field_name] = (field_type, Field(..., **field_info))
                else:
                    fields[field_name] = (Union[field_type, None], Field(None, **field_info))

        new_class = self._create_model(class_name, description, fields)
        self.created_classes[class_name] = new_class
        return new_class


def test1() -> None:
    """Test basic functionality with a comprehensive example.

    Creates a family model with nested Email and Address classes,
    demonstrates instantiation, validation, and error handling.
    """
    test_yaml = """
    Person:
      description: "class for members of the family"
      fields:
        name:
            description: "Person's full name"
            required: true
        age:
            type: int
            description: "Age in years"
            required: false
        email:
            type: list[Email]
            description: "Email addresses"
            required: true
        address:
            type: Address
            description: "Home address"
            required: false

    Email:
      description: "email contact"
      fields:
        url:
            type: str
            required: true
        email_type:
            type: str
            required: false

    Address:
      description: "postal contact"
      fields:
        street:
            type: str
            required: true
        city:
            type: str
            required: true
        zip_code:
            type: str
            required: false
        country:
            type: str
            required: false
    """

    # Test with string content
    converter = PydanticModelFactory()
    yaml_data = yaml.safe_load(test_yaml)

    PersonClass = converter.create_class_from_dict(yaml_data, "Person")

    # Create instances
    person_data = {
        "name": "John Doe",
        "age": 30,
        "email": [{"url": "john@example.com"}, {"url": "myssf@gmail.com", "email_type": "pro"}],
        "address": {"street": "123 Main St", "city": "Anytown", "zip_code": "12345"},
    }

    person = PersonClass(**person_data)
    print("Created person:", person)

    # Test validation
    try:
        _ = PersonClass(name="Jane")
    except Exception as e:
        print("Validation error (expected):", str(e))

    print("All tests completed successfully!")


def test2() -> None:
    from src.demos.ekg.struct_rag_doc_processing import get_schema

    demo = get_schema("test")
    assert demo

    converter = PydanticModelFactory()
    PersonClass = converter.create_class_from_dict(demo["schema"], demo["top_class"])
    print(PersonClass)

    person_data = {
        "name": "John Doe",
        "age": 30,
        "email": [{"url": "john@example.com"}, {"url": "myssf@gmail.com", "email_type": "pro"}],
        "address": {"street": "123 Main St", "city": "Anytown", "zip_code": "12345"},
    }

    person = PersonClass(**person_data)
    person.extra_field = "extra field value"  # type: ignore
    person.__setattr__("another_extra_field", 25)
    print("Created person:", person)


def test3() -> None:
    from src.demos.ekg.struct_rag_doc_processing import get_schema

    demo = get_schema("test")
    assert demo

    converter = PydanticModelFactory()
    PersonClass = converter.create_class_from_dict(demo["schema"], demo["top_class"])
    print(PersonClass)

    person_data = {
        "name": "John Doe",
        "age": 30,
        "email": [{"url": "john@example.com"}, {"url": "myssf@gmail.com", "email_type": "pro"}],
        "address": {"street": "123 Main St", "city": "Anytown", "zip_code": "12345"},
    }

    person = PersonClass(**person_data)
    person.extra_field = "extra field value"  # type: ignore
    person.__setattr__("another_extra_field", 25)
    print("Created person:", person)


if __name__ == "__main__":
    test1()
    test2()
