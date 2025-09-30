from typing import Any, get_args


def describe_container_type(obj: Any) -> str:
    """
    Return a readable description of a container's type and its element types.

    Examples:
        >>> describe_container_type([{"a": 1}, {"b": 2}])
        "list[dict]"
        >>> describe_container_type(["hello", "world"])
        "list[str]"
        >>> describe_container_type({1, 2, 3})
        "set[int]"
        >>> describe_container_type({"x": [1, 2, 3]})
        "dict[str, list[int]]"
        >>> describe_container_type(42)
        "int"
    """
    if obj is None:
        return "None"

    # Handle basic scalar types
    if not isinstance(obj, (list, tuple, set, dict)):
        return type(obj).__name__

    container_type = type(obj).__name__

    # Empty containers
    if not obj:
        # Try to get generic type args if available
        if hasattr(obj, "__orig_class__"):
            type_args = get_args(obj.__orig_class__)  # pyright: ignore[reportAttributeAccessIssue]
            if type_args:
                arg_str = ", ".join(arg.__name__ for arg in type_args)
                return f"{container_type}[{arg_str}]"
        return f"{container_type}[Any]"

    # Non-empty containers
    if isinstance(obj, dict):
        key_types = {type(k).__name__ for k in obj.keys()}
        value_types = {type(v).__name__ for v in obj.values()}

        key_type = next(iter(key_types)) if len(key_types) == 1 else "Union[" + ",".join(sorted(key_types)) + "]"
        value_type = (
            next(iter(value_types)) if len(value_types) == 1 else "Union[" + ",".join(sorted(value_types)) + "]"
        )

        return f"{container_type}[{key_type}, {value_type}]"

    elif isinstance(obj, (list, tuple, set)):
        element_types = {type(item).__name__ for item in obj}
        element_type = (
            next(iter(element_types)) if len(element_types) == 1 else "Union[" + ",".join(sorted(element_types)) + "]"
        )

        return f"{container_type}[{element_type}]"

    return str(type(obj))


def test_container_types():
    """Test the describe_container_type function with various examples."""
    test_cases = [
        [1, 2, 3],
        ["a", "b", "c"],
        [{"x": 1}, {"y": 2}],
        [{"x": 1}, "mixed"],
        [],
        {},
        {"a": 1, "b": 2},
        {"nested": {"deep": "value"}},
        {"list": [1, 2, 3]},
        {1, 2, 3},
        (1, 2, 3),
        42,
        "string",
        None,
    ]

    for case in test_cases:
        print(f"{repr(case)} -> {describe_container_type(case)}")


if __name__ == "__main__":
    test_container_types()
