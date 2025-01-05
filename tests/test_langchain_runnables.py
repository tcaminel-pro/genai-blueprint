import pytest
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)
from operator import itemgetter
from loguru import logger

# Utility functions from the notebook
def mult_2(x: int):
    return x * 2

def add_1(x: int):
    return x + 1

def add_3(x: int):
    return x + 3

def test_runnable_lambda():
    """Test basic RunnableLambda functionality"""
    add_1_runnable = RunnableLambda(add_1)
    assert add_1_runnable.invoke(5) == 6

def test_runnable_sequence():
    """Test sequential runnable composition"""
    sequence = mult_2 | add_1
    assert sequence.invoke(1) == 3
    
    # Batch processing
    batch_result = sequence.batch([1, 2, 3])
    assert batch_result == [3, 5, 7]

def test_runnable_parallel():
    """Test parallel runnable composition"""
    parallel = mult_2 | {
        "add_1": RunnableLambda(add_1), 
        "add_3": RunnableLambda(add_3)
    }
    
    result = parallel.invoke(1)
    assert result == {"add_1": 3, "add_3": 4}

def test_runnable_passthrough():
    """Test RunnablePassthrough functionality"""
    runnable = RunnableParallel(
        origin=RunnablePassthrough(), 
        modified=RunnableLambda(add_1)
    )
    
    result = runnable.invoke(10)
    assert result == {"origin": 10, "modified": 11}

def test_runnable_max_filter():
    """Test runnable with max filter and logging"""
    @chain
    def max_filter(x: int, max: int, config: dict = None) -> int:
        return max if x >= max else x
    
    sequence = (mult_2 | add_1) | max_filter.bind(max=6)
    
    # Test individual invocation
    assert sequence.invoke(1) == 3
    
    # Test batch processing with logging
    batch_result = sequence.batch([1, 2, 3, 4, 5], config=[{"logger": logger}])
    assert batch_result == [3, 5, 6, 6, 6]

def test_runnable_assign():
    """Test runnable with assign method"""
    runnable = (
        RunnableParallel(
            extra=RunnablePassthrough.assign(mult_10=lambda x: x["num"] * 10),
            plus_1=lambda x: x["num"] + 1,
        )
        .assign(info=lambda x: x)
        .assign(plus_1_time_3=lambda x: x["plus_1"] * 3)
    )
    
    result = runnable.invoke({"num": 2})
    assert result == {
        "extra": {"num": 2, "mult_10": 20},
        "plus_1": 3,
        "info": {"num": 2, "plus_1": 3, "extra": {"num": 2, "mult_10": 20}},
        "plus_1_time_3": 9
    }

def test_runnable_itemgetter():
    """Test runnable with itemgetter"""
    adder = RunnableLambda(lambda d: d["op1"] + d["op2"])
    
    mult_2_and_add = (
        RunnableParallel(
            op1=itemgetter("a") | mult_2,
            op2=itemgetter("b") | mult_2,
        )
        | adder
    )
    
    result = mult_2_and_add.invoke({"a": 10, "b": 2, "z": "sds"})
    assert result == 24

def test_runnable_fallback():
    """Test runnable with fallback mechanism"""
    @chain
    def mult_10_fail(x: int):
        raise Exception("unavailable multiplication by 10 service")
    
    fallback_chain = mult_10_fail.with_fallbacks([mult_2])
    
    result = fallback_chain.invoke(2)
    assert result == 4
