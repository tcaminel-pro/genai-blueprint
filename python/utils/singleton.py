"""
Implement 'once', a decorator that ensures the wrapped function is called once and return same result.\n
It's typically used for thread-safe singleton instance creation.

"""

import inspect
from functools import wraps
from threading import Lock


def once():
    """
    A decorator that ensures the wrapped function is called once and return same result.\n
    It's typically used for thread-safe singleton instance creation.

    It's inspired by the 'once' keyword in the Eiffel Programming language.
    It's simpler and arguably clearer than most usual approach to create singletons, such as inheriting a metaclass, overriding __init__(), etc.

    Purists might say it's not a 'real' Singleton class (as defined by the GoF), we can argue that it actually enforce reusability,
    since the class has not to be specialized to become a singleton.

    Example:
        @once()
        def get_instance(x: int):
            return MyClass(x)

        @once()
        def singleton() -> "MyClass":
            "Returns a singleton instance of the class"
            return MyClass()

        my_class_singleton = MyClass.singleton()
    """

    def decorator(func):
        # Having more than 1 parameter in the function is not fully implemented, so we forbit that case
        params = inspect.signature(func).parameters
        if len(params) > 1:
            raise ValueError(
                f"'once' function cannot have more that 1 parameter, but '{func.__name__}' has: {tuple(params.keys())}"
            )

        setattr(decorator, "_cached_results", {})  # Store instance and lock as decorator attributes
        setattr(decorator, "_lock", Lock())

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on the arguments - But does not work well with several arguments
            sorted_kwargs = tuple(sorted(kwargs.items()))
            cache_key = (args, sorted_kwargs)

            if cache_key not in getattr(decorator, "_cached_results"):
                with getattr(decorator, "_lock"):
                    if cache_key not in getattr(decorator, "_cached_results"):
                        result = func(*args, **kwargs)
                        getattr(decorator, "_cached_results")[cache_key] = result
            return getattr(decorator, "_cached_results")[cache_key]

        return wrapper

    return decorator


# TEST
if __name__ == "__main__":
    from pydantic import BaseModel, ConfigDict

    class MyClass(BaseModel):
        model_config = ConfigDict(frozen=True)
        a: int
        b: int = 1

        @once()
        def singleton() -> "MyClass":
            """Returns a singleton instance of the class"""
            return MyClass(a=1)

    # Usage example:
    obj1 = MyClass.singleton()
    obj2 = MyClass.singleton()
    assert obj1 is obj2  # True - same instance

    @once()
    def get_my_class_singleton():
        return MyClass(a=4)

    obj3 = get_my_class_singleton()
    obj4 = get_my_class_singleton()
    assert obj3 is obj4  # True - same instance

    assert obj1 is not obj3

    @once()
    def do_something_complicated(x: int):
        return MyClass(a=x)

    obj5 = do_something_complicated(1)
    obj6 = do_something_complicated(1)
    obj7 = do_something_complicated(2)

    assert obj5 is obj6
    assert obj5 is not obj7
    assert obj7.a == 2

    @once()
    def do_something_complicated_2(x: int, y: int):  # fail
        return MyClass(a=x + y)
