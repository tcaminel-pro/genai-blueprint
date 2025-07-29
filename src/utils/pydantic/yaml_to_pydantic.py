"""Create Pydantic classes dynamically from YAML definitions."""

from typing import Any, Dict, List, Type, Union

import yaml
from pydantic import BaseModel, Field, create_model


def yaml_type_to_python_type(yaml_type: str) -> type:
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
        return List[yaml_type_to_python_type(inner_type)]
    elif yaml_type.startswith("Dict[") or yaml_type.startswith("dict["):
        inner_types = yaml_type[5:-1].split(",")
        key_type = yaml_type_to_python_type(inner_types[0].strip())
        value_type = yaml_type_to_python_type(inner_types[1].strip())
        return Dict[key_type, value_type]

    return type_mapping.get(yaml_type, str)


def load_yaml_and_create_class(yaml_content: str, class_name: str | None = None) -> Type[BaseModel]:
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

    created_classes: Dict[str, Type[BaseModel]] = {}

    def create_class(class_name: str, class_def: Dict[str, Any]) -> Type[BaseModel]:
        if class_name in created_classes:
            return created_classes[class_name]

        fields = {}

        for field_name, field_def in class_def.items():
            if isinstance(field_def, dict):
                if "type" in field_def and field_def["type"] in yaml_data:
                    nested_class_name = field_def["type"]
                    if nested_class_name not in created_classes:
                        if nested_class_name in yaml_data:
                            create_class(nested_class_name, yaml_data[nested_class_name])

                    field_type = created_classes[nested_class_name]
                    is_required = field_def.get("required", True)
                else:
                    yaml_type = field_def.get("type", "str")
                    field_type = yaml_type_to_python_type(yaml_type)
                    is_required = field_def.get("required", True)

                field_info = {}
                if "description" in field_def:
                    field_info["description"] = field_def["description"]

                if is_required:
                    fields[field_name] = (field_type, Field(..., **field_info))
                else:
                    fields[field_name] = (Union[field_type, None], Field(None, **field_info))

        new_class = create_model(class_name, __base__=BaseModel, **fields)

        created_classes[class_name] = new_class
        return new_class

    for cls_name, cls_def in yaml_data.items():
        if cls_name not in created_classes:
            create_class(cls_name, cls_def)

    if class_name:
        return created_classes[class_name]

    return list(created_classes.values())[0]


if __name__ == "__main__":
    # Test the functionality
    test_yaml = """
    Person:
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
      url:
        type: str
        description: "URL"
      email_type:
        type: str
        description: "personal or professional"

    Address:
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
    PersonClass = load_yaml_and_create_class(test_yaml, "Person")

    # Create instances
    person_data = {
        "name": "John Doe",
        "age": 30,
        "email": [
            {"url": "john@example.com", "email_type": "pro"}, 
            {"url": "myssf@gmail.com", "email_type": "pro"}
        ],
        "address": {"street": "123 Main St", "city": "Anytown", "zip_code": "12345", "country": "USA"},
    }

    from rich import print

    person = PersonClass(**person_data)
    print("Created person:", person)

    # Test validation
    try:
        invalid_person = PersonClass(name="Jane")
    except Exception as e:
        print("Validation error (expected):", str(e))

    print("All tests completed successfully!")
