"""Create Pydantic classes dynamically from YAML definitions."""

from typing import Any, Dict, List, Type, Union

import yaml
from pydantic import BaseModel, Field, create_model
from rich import print


class YamlToPydantic:
    """Convert YAML definitions to Pydantic classes dynamically."""

    def __init__(self) -> None:
        """Initialize the YamlToPydantic converter."""
        self.created_classes: Dict[str, Type[BaseModel]] = {}

    def yaml_type_to_python_type(self, yaml_type: str) -> type:
        """Convert YAML type string to Python type."""
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

        if yaml_type.startswith("List[") or yaml_type.startswith("list["):
            inner_type = yaml_type[5:-1]
            return List[self.yaml_type_to_python_type(inner_type)]
        elif yaml_type.startswith("Dict[") or yaml_type.startswith("dict["):
            inner_types = yaml_type[5:-1].split(",")
            key_type = self.yaml_type_to_python_type(inner_types[0].strip())
            value_type = self.yaml_type_to_python_type(inner_types[1].strip())
            return Dict[key_type, value_type]

        return type_mapping.get(yaml_type, str)

    def create_class_from_yaml(self, yaml_content: str, class_name: str | None = None) -> Type[BaseModel]:
        """Create a Pydantic class from YAML content string.

        Args:
            yaml_content: YAML content as string
            class_name: Optional name for the class to return

        Returns:
            A dynamically created Pydantic class
        """
        yaml_data = yaml.safe_load(yaml_content)

        if not yaml_data:
            raise ValueError("Invalid YAML content")

        return self.create_class_from_dict(yaml_data, class_name)

    def create_class_from_dict(self, yaml_data: dict, class_name: str | None = None) -> Type[BaseModel]:
        """Create a Pydantic class from dictionary data.

        Args:
            yaml_data: Dictionary containing class definitions
            class_name: Optional name for the class to return

        Returns:
            A dynamically created Pydantic class
        """
        self.created_classes.clear()

        # First pass: create all class definitions
        for cls_name, cls_def in yaml_data.items():
            if cls_name not in self.created_classes:
                # Create placeholder classes to handle circular dependencies
                self.created_classes[cls_name] = None

        # Second pass: create actual classes
        for cls_name, cls_def in yaml_data.items():
            if cls_name not in self.created_classes or self.created_classes[cls_name] is None:
                self._create_class(cls_name, cls_def, yaml_data)

        if class_name:
            return self.created_classes[class_name]

        return list(self.created_classes.values())[0]

    def _create_model_with_description(self, model_name: str, description: str, fields: dict) -> Type[BaseModel]:
        """Create a Pydantic model with a programmatic description."""
        if description:
            # Create config dict with description
            config_dict = {"description": description}
            return create_model(model_name, __config__=config_dict, __base__=BaseModel, **fields)
        else:
            return create_model(model_name, __base__=BaseModel, **fields)

    def _create_class(self, class_name: str, class_def: Dict[str, Any], yaml_data: dict) -> Type[BaseModel]:
        """Internal method to create a single Pydantic class."""
        if class_name in self.created_classes and self.created_classes[class_name] is not None:
            return self.created_classes[class_name]

        description = class_def.get("description", "")
        fields_def = class_def.get("fields", class_def)

        fields = {}
        for field_name, field_def in fields_def.items():
            if isinstance(field_def, dict):
                yaml_type = field_def.get("type", "str")
                is_required = field_def.get("required", False)

                if yaml_type.startswith("list[") or yaml_type.startswith("List["):
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

        new_class = self._create_model_with_description(class_name, description, fields)
        self.created_classes[class_name] = new_class
        return new_class


def test1() -> None:
    # Test the functionality
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
    converter = YamlToPydantic()
    PersonClass = converter.create_class_from_yaml(test_yaml, "Person")

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
    from src.utils.config_mngr import global_config

    demo = (
        global_config()
        .merge_with("config/demos/document_extractor.yaml")
        .get_dict("Document_extractor_demo.0", expected_keys=["schema", "key", "top_class"])
    )

    converter = YamlToPydantic()
    PersonClass = converter.create_class_from_dict(demo["schema"], demo["top_class"])
    print(PersonClass)

    person_data = {
        "name": "John Doe",
        "age": 30,
        "email": [{"url": "john@example.com"}, {"url": "myssf@gmail.com", "email_type": "pro"}],
        "address": {"street": "123 Main St", "city": "Anytown", "zip_code": "12345"},
    }

    person = PersonClass(**person_data)
    print("Created person:", person)


if __name__ == "__main__":
    test1()
    test2()
