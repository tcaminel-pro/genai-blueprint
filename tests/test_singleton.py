""" "
Tests for the singleton.py module
"""

import pytest
from pydantic import BaseModel, ConfigDict

from python.utils.singleton import once


class TestModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: int

    @once()
    def singleton() -> "TestModel":
        """Returns a singleton instance of the class"""
        return TestModel(value=42)


@once()
def test_singleton_func() -> int:
    """Test singleton function"""
    return 100


def test_class_singleton():
    """Test that class method returns same instance"""
    instance1 = TestModel.singleton()
    instance2 = TestModel.singleton()

    assert instance1 is instance2
    assert instance1.value == 42


    assert instance2.value == 42


def test_function_singleton():
    """Test that function returns same value"""
    val1 = test_singleton_func()
    val2 = test_singleton_func()

    assert val1 is val2
    assert val1 == 100


def test_singleton_with_args():
    """Test that functions with args work with caching"""
    @once()
    def cached_func(x: int, y: int = 0):
        return TestModel(value=x + y)

    # Same args - same instance
    instance1 = cached_func(1, 2)
    instance2 = cached_func(1, 2)
    assert instance1 is instance2
    assert instance1.value == 3

    # Different args - different instances
    instance3 = cached_func(2, 3)
    assert instance3 is not instance1
    assert instance3.value == 5

    # Test default args
    instance4 = cached_func(1)
    assert instance4 is not instance1
    assert instance4.value == 1


def test_different_singletons():
    """Test that different singletons return different instances"""
    class_instance = TestModel.singleton()
    func_value = test_singleton_func()

    assert class_instance is not func_value


def test_thread_safety():
    """Test that singleton creation is thread-safe"""
    import threading

    results = []
    lock = threading.Lock()

    @once()
    def thread_test_func():
        return TestModel(value=42)

    def worker():
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
