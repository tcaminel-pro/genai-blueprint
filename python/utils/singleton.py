"""
Implement 'once', a decorator that ensures the wrapped function is called once and return same result.\n
It's typically used for thread-safe singleton instance creation.

"""

import inspect
from functools import wraps
from threading import Lock


def once(with_args: bool = True):
    """
    A decorator that ensures the wrapped function is called once and return same result.\n
    It's typically used for thread-safe singleton instance creation.

    It's inspired by the 'once' keyword in the Eiffel Programming language.
    It's simpler and arguably clearer than most usual approach to create singletons, such as inheriting a metaclass, overriding __init__(), etc.

    Purists might say it's not a 'real' Singleton class (as defined by the GoF), we can argue that it actually enforce reusability,
    since the class has not to be specialized to become a singleton.

    example use:
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

    # How to pass an argument to an operator (here :with_args - does not work ) AI?
    def decorator(func):
        params = inspect.signature(func).parameters
        if len(params) > 0 and not with_args:
            raise ValueError(
                f"'once' function cannot have parameters if 'with_args' is False, but '{func.__name__}' has: {tuple(params.keys())}"
            )

        setattr(decorator, "_cached_results", {})  # Store instance and lock as decorator attributes
        setattr(decorator, "_lock", Lock())

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on the arguments
            # Sort kwargs items to handle different argument orders consistently
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
    def do_something_complicated(x: int, y: int):
        return MyClass(a=x + y)

    obj5 = do_something_complicated(1, 2)
    obj6 = do_something_complicated(1, 2)
    obj7 = do_something_complicated(2, 3)

    assert obj5 is obj6
    assert obj5 is not obj7
    assert obj7.a == 5

    # @once(with_args=False)
    # def do_something_complicated_2(x: int, y: int):
    #     return MyClass(a=x + y)

    # obj5 = do_something_complicated_2(1, 2)
    # obj6 = do_something_complicated_2(1, 2)
    # obj7 = do_something_complicated_2(2, 3)
