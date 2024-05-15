from typing import Any, Type, TypeVar

from pydantic import BaseModel, create_model

T = TypeVar("T", bound=BaseModel)


def add_field_to_class(
    base_class: Type[T], field_name: str, field_type: Type, required: bool = False
) -> Type[T]:
    """
    Creates a new Pydantic class by inheriting from the provided base class
    and adding a new field with the specified name and type.

    Args:
        base_class (Type[T]): The base Pydantic class to inherit from.
        field_name (str): The name of the new field to add.
        field_type (Type): The type of the new field.
        required (bool, optional): Whether the new field should be required or not. Defaults to False.

    Returns:
        Type[T]: A new Pydantic class with the additional field.
    """
    field_definition = (field_type, ...) if required else (field_type, None)
    new_class_name = f"{base_class.__name__}With{field_name.capitalize()}"

    return create_model(
        new_class_name,
        **{field_name: field_definition},  # type: ignore
        __base__=base_class,
    )  # type: ignore


def add_field_to_obj(obj: BaseModel, field_name: str, value: Any) -> BaseModel:
    """
    Creates a new Pydantic object having a descendant class and a new field with the specified name and type.
    """

    new_cls = add_field_to_class(type(obj), field_name, type(value), True)
    new_obj = new_cls.model_validate(obj.model_dump() | {field_name: value})
    return new_obj


def test():
    class Car(BaseModel):
        brand: str
        model: str
        year: int

    CarWithColor = add_field_to_class(Car, "color", str, required=True)
    my_car = CarWithColor(brand="Toyota", model="Camry", year=2022, color="Red")
    new_car = add_field_to_obj(my_car, "id", "12345")
    print(new_car)


if __name__ == "__main__":
    test()
