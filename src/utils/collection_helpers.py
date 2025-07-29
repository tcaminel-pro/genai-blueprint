"""Collection helper utilities for more readable collection operations."""

from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


def find_first(items: Iterable[T], predicate: Callable[[T], bool]) -> T | None:
    """Find the first item in a collection that matches a predicate.

    Args:
        items: The collection to search
        predicate: A function that returns True for the desired item

    Returns:
        The first matching item, or None if no match is found

    Example:
        ```python
        from dataclasses import dataclass

        @dataclass
        class User:
            name: str
            age: int

        users = [User("Alice", 30), User("Bob", 25), User("Charlie", 35)]

        # Find first user named Bob
        bob = find_first(users, lambda u: u.name == "Bob")
        print(bob)  # User(name='Bob', age=25)

        # Find first user over 30
        over_30 = find_first(users, lambda u: u.age > 30)
        print(over_30)  # User(name='Charlie', age=35)

        # No match returns None
        missing = find_first(users, lambda u: u.name == "David")
        print(missing)  # None
        ```
    """
    return next((item for item in items if predicate(item)), None)


def find_by_key(items: Iterable[T], key: str, value: Any) -> T | None:
    """Find the first item in a collection where a specific key matches a value.

    Args:
        items: The collection to search
        key: The attribute name to check
        value: The value to match

    Returns:
        The first matching item, or None if no match is found

    Example:
        ```python
        from dataclasses import dataclass

        @dataclass
        class Product:
            id: int
            name: str
            category: str

        products = [
            Product(1, "Laptop", "Electronics"),
            Product(2, "Chair", "Furniture"),
            Product(3, "Phone", "Electronics")
        ]

        # Find product by name
        laptop = find_by_key(products, 'name', 'Laptop')
        print(laptop)  # Product(id=1, name='Laptop', category='Electronics')

        # Find product by category
        furniture = find_by_key(products, 'category', 'Furniture')
        print(furniture)  # Product(id=2, name='Chair', category='Furniture')

        # No match returns None
        missing = find_by_key(products, 'name', 'Tablet')
        print(missing)  # None
        ```
    """
    return next((item for item in items if getattr(item, key, None) == value), None)
