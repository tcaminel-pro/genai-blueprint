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
    It's simpler and clearer than most usual approach to create singletons, such as inheriting a metaclass, overriding __init__(), etc.

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

    def decorator(func):
        # check the function has no parameters
        if len(inspect.signature(func).parameters) > 0:
            print(inspect.signature(func).parameters, func)
            # raise ValueError("'once' function cannot have parameters (in current version - might be introduced later)")

        # Store instance and lock as decorator attributes
        setattr(decorator, "_cached_result", None)
        setattr(decorator, "_lock", Lock())

        @wraps(func)
        def wrapper(*args, **kwargs):
            if getattr(decorator, "_cached_result") is None:
                with getattr(decorator, "_lock"):  # acquire the lock for thread-safety
                    if getattr(decorator, "_cached_result") is None:
                        setattr(decorator, "_cached_result", func(*args[1:], **kwargs))
            return getattr(decorator, "_cached_result")

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
        @staticmethod
        def singleton() -> "MyClass":
            """Returns a singleton instance of the class"""
            return MyClass(a=1)

    # Usage example:
    obj1 = MyClass.singleton()
    obj2 = MyClass.singleton()
    # obj2.a = 2

    assert obj1 is obj2  # True - same instance

    # obj3 = MyClass(a=4)
    # obj4 = MyClass(a=4)
    # assert obj1 is not obj3
    # assert obj4 is not obj3
    # debug(obj1)

    x: int = 0

    @once()
    def inc(a):
        global x
        print("execute code")
        x = x + 1
        return x

    print(inc(2))  # should be 1
    print(inc(2))  # should be 1
    print(inc(2))  # should be 1
