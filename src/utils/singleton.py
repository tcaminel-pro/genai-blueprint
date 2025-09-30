"""
Implement 'once', a decorator that ensures the wrapped function is called once and return same result.\n
It's typically used for thread-safe singleton instance creation.

It's inspired by the 'once' keyword in the Eiffel Programming language.
It's simpler and arguably clearer than most usual approach to create singletons, such as inheriting a metaclass,
overriding __init__(), etc.

Purists might say it's not a 'real' Singleton class (as defined by the GoF), but  we can argue that
it actually enforce reusability, since the class has not to be specialized to become a singleton.

"""

from __future__ import annotations

import inspect
from functools import wraps
from threading import Lock
from typing import Any, Callable, TypeVar

R = TypeVar("R")


def once(func: Callable[..., R]) -> Callable[..., R]:
    """
    A decorator that ensures the wrapped function is called once and return same result.\n
    It's typically used for thread-safe singleton instance creation.

    Example:
    ```
        class MyClass (BaseModel):
            model_config = ConfigDict(frozen=True)

            @once
            def singleton() -> "MyClass":
                "Returns a singleton instance of the class"
                return MyClass()

        my_class_singleton = MyClass.singleton()

        # Invalidate the singleton
        MyClass.singleton.invalidate()
        new_instance = MyClass.singleton()  # Creates a new instance

    # work for functions, too:
        @once
        def get_my_class_singleton():
            return MyClass()

        # Invalidate function-based singleton
        get_my_class_singleton.invalidate()
    ```
    """
    # Preserve the original function's docstring and attributes
    original_doc = func.__doc__
    original_annotations = func.__annotations__
    if isinstance(func, staticmethod):
        # If already a staticmethod, apply once_fn() to the underlying function
        inner_func = func.__func__
        wrapper = once_fn()(inner_func)
        wrapped = staticmethod(wrapper)
        # Forward the invalidate method to the staticmethod
        wrapped.invalidate = wrapper.invalidate
    else:
        # Otherwise create a new staticmethod with once_fn() applied
        wrapper = once_fn()(func)
        wrapped = staticmethod(wrapper)
        # Expose the invalidate method directly
        wrapped.invalidate = wrapper.invalidate

    # Preserve the original function's documentation and attributes
    wrapped.__doc__ = original_doc
    wrapped.__annotations__ = original_annotations
    if isinstance(func, staticmethod):
        wrapped.__name__ = inner_func.__name__
        wrapped.__module__ = inner_func.__module__
    else:
        wrapped.__name__ = func.__name__
        wrapped.__module__ = func.__module__

    return wrapped


def once_fn() -> Callable:
    """Factory function that returns a decorator for once functionality."""

    def decorator(func: Callable):
        """The actual decorator that implements the once functionality."""
        decorator._cached_results = {}  # type: ignore # Store instance and lock as decorator attributes
        decorator._lock = Lock()  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a stable cache key that handles:
            # - Multiple arguments
            # - Mutable types (lists, dicts)
            # - None values
            # - Keyword argument order
            def make_hashable(obj) -> Any:  # noqa: ANN001
                if obj is None:
                    return None
                if isinstance(obj, (int, float, str, bool)):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return tuple(make_hashable(x) for x in obj)
                if isinstance(obj, dict):
                    return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
                if hasattr(obj, "__dict__"):
                    return make_hashable(vars(obj))
                return str(obj)

            # Get function signature to properly handle default args
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Create cache key that treats positional and keyword args the same
            sorted_args = tuple(sorted((k, make_hashable(v)) for k, v in bound_args.arguments.items()))
            cache_key = sorted_args

            if cache_key not in decorator._cached_results:  # type: ignore
                with decorator._lock:  # type: ignore
                    if cache_key not in decorator._cached_results:  # type: ignore
                        result = func(*args, **kwargs)
                        decorator._cached_results[cache_key] = result  # type: ignore
            return decorator._cached_results[cache_key]  # type: ignore

        # Add invalidation method
        def invalidate() -> None:
            with decorator._lock:  # type: ignore
                decorator._cached_results.clear()  # type: ignore

        wrapper.invalidate = invalidate

        return wrapper

    return decorator  # type: ignore


# TEST


if __name__ == "__main__":
    from pydantic import BaseModel, ConfigDict

    class TestClass1(BaseModel):
        model_config = ConfigDict(frozen=True)
        a: int
        b: int = 1

        @once
        def singleton() -> TestClass1:
            """Returns a singleton instance of the class"""
            return TestClass1(a=1)

        @once
        def singleton2(a: int, b: int) -> TestClass1:  # type: ignore
            """Returns a singleton instance of the class"""
            return TestClass1(a=a, b=b)

    # Usage example:
    obj1 = TestClass1.singleton()
    obj2 = TestClass1.singleton()
    assert obj1 is obj2  # True - same instance

    @once
    def get_my_class_singleton() -> TestClass1:
        return TestClass1(a=4)

    obj3 = get_my_class_singleton()
    obj4 = get_my_class_singleton()
    assert obj3 is obj4  # True - same instance

    assert obj1 is not obj3

    @once
    def do_something(x: int) -> TestClass1:
        return TestClass1(a=x)

    obj5 = do_something(1)
    obj6 = do_something(1)
    obj7 = do_something(2)

    assert obj5 is obj6
    assert obj5 is not obj7
    assert obj7.a == 2

    @once
    def do_something_2(x: int, y: int) -> TestClass1:
        return TestClass1(a=x + y)

    # Test multiple arguments
    obj8 = do_something_2(1, 2)
    obj9 = do_something_2(1, 2)
    obj10 = do_something_2(2, 1)

    assert obj8 is obj9  # Same args - same instance
    assert obj8 is not obj10  # Different args - different instance
    assert obj8.a == 3
    assert obj10.a == 3  # Same sum but different args

    # Test invalidation
    obj14 = TestClass1.singleton()
    TestClass1.singleton.invalidate()
    obj15 = TestClass1.singleton()
    assert obj14 is not obj15

    # Test function-based invalidation
    obj16 = get_my_class_singleton()
    get_my_class_singleton.invalidate()
    obj17 = get_my_class_singleton()
    assert obj16 is not obj17

    obj11 = TestClass1.singleton2(a=1, b=2)
    obj12 = TestClass1.singleton2(a=1, b=2)
    obj13 = TestClass1.singleton2(a=3, b=4)
    assert obj11 is obj12
    assert obj13 is not obj11
