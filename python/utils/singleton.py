"""
Implement 'once', a decorator that ensures the wrapped function is called once and return same result.\n
It's typically used for thread-safe singleton instance creation.

It's inspired by the 'once' keyword in the Eiffel Programming language.
It's simpler and arguably clearer than most usual approach to create singletons, such as inheriting a metaclass, overriding __init__(), etc.

Purists might say it's not a 'real' Singleton class (as defined by the GoF), we can argue that it actually enforce reusability,
since the class has not to be specialized to become a singleton.

"""

import inspect
from functools import wraps
from threading import Lock
from typing import Any

# add typying annotations whenever possible AI!

def once():
    """
    A decorator that ensures the wrapped function is called once and return same result.\n
    It's typically used for thread-safe singleton instance creation.

    Example:
    ```
        class MyClass (BaseModel):
            model_config = ConfigDict(frozen=True)

            @once()
            def singleton() -> "MyClass":
                "Returns a singleton instance of the class"
                return MyClass()

        my_class_singleton = MyClass.singleton()

    # work for functions, too:
        @once()
        def get_my_class_singleton():
            return MyClass()
            ...
    """

    def decorator(func):
        decorator._cached_results = {}  # type: ignore # Store instance and lock as decorator attributes
        decorator._lock = Lock()  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create a stable cache key that handles:
            # - Multiple arguments
            # - Mutable types (lists, dicts)
            # - None values
            # - Keyword argument order
            def make_hashable(obj) -> Any:
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

            if cache_key not in decorator._cached_results:
                with decorator._lock:
                    if cache_key not in decorator._cached_results:
                        result = func(*args, **kwargs)
                        decorator._cached_results[cache_key] = result
            return decorator._cached_results[cache_key]

        return wrapper

    return decorator


# TEST
if __name__ == "__main__":
    from pydantic import BaseModel, ConfigDict

    class SingletonTestModel(BaseModel):
        model_config = ConfigDict(frozen=True)
        a: int
        b: int = 1

        @once()
        def singleton() -> "SingletonTestModel":
            """Returns a singleton instance of the class"""
            return SingletonTestModel(a=1)

    # Usage example:
    obj1 = SingletonTestModel.singleton()
    obj2 = SingletonTestModel.singleton()
    assert obj1 is obj2  # True - same instance

    @once()
    def get_my_class_singleton() -> SingletonTestModel:
        return SingletonTestModel(a=4)

    obj3 = get_my_class_singleton()
    obj4 = get_my_class_singleton()
    assert obj3 is obj4  # True - same instance

    assert obj1 is not obj3

    @once()
    def do_something_complicated(x: int) -> SingletonTestModel:
        return SingletonTestModel(a=x)

    obj5 = do_something_complicated(1)
    obj6 = do_something_complicated(1)
    obj7 = do_something_complicated(2)

    assert obj5 is obj6
    assert obj5 is not obj7
    assert obj7.a == 2

    @once()
    def do_something_complicated_2(x: int, y: int) -> SingletonTestModel:
        return SingletonTestModel(a=x + y)

    # Test multiple arguments
    obj8 = do_something_complicated_2(1, 2)
    obj9 = do_something_complicated_2(1, 2)
    obj10 = do_something_complicated_2(2, 1)

    assert obj8 is obj9  # Same args - same instance
    assert obj8 is not obj10  # Different args - different instance
    assert obj8.a == 3
    assert obj10.a == 3  # Same sum but different args
