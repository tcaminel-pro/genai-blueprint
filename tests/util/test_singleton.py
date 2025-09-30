""" "
Tests for the singleton.py module.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.utils.singleton import once


class SingletonExample(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: int

    @once
    def singleton() -> SingletonExample:
        """Returns a singleton instance of the class."""
        return SingletonExample(value=42)


@once
def singleton_func() -> int:
    """Test singleton function."""
    return 100


def test_class_singleton() -> None:
    """Test that class method returns same instance."""
    instance1 = SingletonExample.singleton()
    instance2 = SingletonExample.singleton()

    assert instance1 is instance2
    assert instance1.value == 42

    assert instance2.value == 42


def test_function_singleton() -> None:
    """Test that function returns same value."""
    val1 = singleton_func()
    val2 = singleton_func()

    assert val1 is val2
    assert val1 == 100


def test_singleton_with_args() -> None:
    @once
    def do_something_complicated(x: int, y: int):
        return SingletonExample(value=x + y)

    obj5 = do_something_complicated(10, 2)
    obj6 = do_something_complicated(10, 2)
    obj7 = do_something_complicated(2, 3)

    assert obj5 is obj6
    assert obj5 is not obj7
    assert obj7.value == 5

    """Test that functions with args work with caching"""

    @once
    def cached_func(x: int, y: int = 0):
        return SingletonExample(value=x + y)

    # Same args - same instance
    instance1 = cached_func(1, 4)
    instance2 = cached_func(1, 4)
    assert instance1 is instance2
    assert instance1.value == 5

    # Different args - different instances
    instance3 = cached_func(4, 1)
    assert instance3 is not instance1
    assert instance3.value == 5

    # Test default args
    instance4 = cached_func(1)
    assert instance4 is not instance1
    assert instance4.value == 1

    # Test keyword args order doesn't matter
    instance5 = cached_func(y=4, x=1)
    assert instance5 is instance1

    # Test mutable args are handled correctly
    @once
    def cached_list_func(items: list):
        return SingletonExample(value=sum(items))

    list1 = [1, 2, 3]
    instance6 = cached_list_func(list1)
    instance7 = cached_list_func(list1.copy())
    assert instance6 is instance7

    # Changing the list shouldn't affect cached result
    list1.append(4)
    instance8 = cached_list_func(list1)
    assert instance8 is not instance6
    assert instance8.value == 10

    # Test multiple args with mutable types
    @once
    def multi_arg_func(a: int, b: list, c: dict):
        return SingletonExample(value=a + sum(b) + sum(c.values()))

    list2 = [1, 2]
    dict1 = {"x": 3}
    instance9 = multi_arg_func(1, list2, dict1)
    instance10 = multi_arg_func(1, list2.copy(), dict1.copy())
    assert instance9 is instance10
    assert instance9.value == 7

    # Different args should create new instance
    list2.append(3)
    instance11 = multi_arg_func(1, list2, dict1)
    assert instance11 is not instance9
    assert instance11.value == 10

    # Test with None values
    @once
    def none_arg_func(a: int | None, b: str | None = None):
        return SingletonExample(value=a if a else 0)

    instance12 = none_arg_func(None)
    instance13 = none_arg_func(None)
    assert instance12 is instance13
    assert instance12.value == 0

    instance14 = none_arg_func(5)
    assert instance14 is not instance12
    assert instance14.value == 5


def test_different_singletons() -> None:
    """Test that different singletons return different instances."""
    class_instance = SingletonExample.singleton()
    func_value = singleton_func()

    assert class_instance is not func_value


def test_thread_safety() -> None:
    """Test that singleton creation is thread-safe."""
    import threading

    results = []
    lock = threading.Lock()

    @once
    def thread_test_func():
        return SingletonExample(value=42)

    def worker() -> None:
        instance = thread_test_func()
        with lock:
            results.append(instance)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should get the same instance
    assert all(r is results[0] for r in results)
