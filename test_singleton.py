"""
Tests for the singleton.py module
"""

import pytest
from pydantic import BaseModel, ConfigDict

from python.utils.singleton import once


class TestModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: int

    @once()
    def singleton(cls) -> "TestModel":
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


def test_singleton_with_args_raises():
    """Test that functions with args raise ValueError"""
    with pytest.raises(ValueError):

        @once()
        def invalid_func(x: int):
            pass


def test_different_singletons():
    """Test that different singletons return different instances"""
    class_instance = TestModel.singleton()
    func_value = test_singleton_func()
    
    assert class_instance is not func_value
