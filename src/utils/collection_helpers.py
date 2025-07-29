"""Collection helper utilities for more readable collection operations."""

from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

# provide self contained examples AI!

def find_first(items: Iterable[T], predicate: Callable[[T], bool]) -> T | None:
    """Find the first item in a collection that matches a predicate.

    Args:
        items: The collection to search
        predicate: A function that returns True for the desired item

    Returns:
        The first matching item, or None if no match is found

    Example:
        ```python
        result = find_first(SAMPLES_DEMOS, lambda d: d.name == selected_pill)
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
        result = find_by_key(SAMPLES_DEMOS, 'name', selected_pill)
        ```
    """
    return next((item for item in items if getattr(item, key, None) == value), None)
