"""Record and replay Streamlit actions for testing and demonstration purposes."""

from __future__ import annotations
from typing import Any, List, Tuple, Callable
import streamlit as st
import time

class StreamlitAction:
    """Represents a single Streamlit action with its arguments and timestamp."""
    
    def __init__(self, func: Callable, args: Tuple, kwargs: dict, timestamp: float):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.timestamp = timestamp

class StreamlitRecorder:
    """Records Streamlit actions and allows replaying them."""
    
    def __init__(self, st_module):
        self.st_module = st_module
        self.actions: List[StreamlitAction] = []
        self.last_timestamp: float = None
        self.original_functions = {}
        
    def __enter__(self):
        """Start recording by wrapping Streamlit functions."""
        self._wrap_streamlit_functions()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop recording by restoring original functions."""
        self._unwrap_streamlit_functions()
        
    def _wrap_streamlit_functions(self):
        """Wrap Streamlit functions to record their calls."""
        functions_to_wrap = ['write', 'markdown', 'text', 'header', 'subheader', 'code']
        
        for func_name in functions_to_wrap:
            if hasattr(self.st_module, func_name):
                original_func = getattr(self.st_module, func_name)
                self.original_functions[func_name] = original_func
                
                def make_wrapper(f):
                    def wrapper(*args, **kwargs):
                        # Record the action
                        now = time.time()
                        time_delta = now - self.last_timestamp if self.last_timestamp else 0
                        self.last_timestamp = now
                        self.actions.append(StreamlitAction(f, args, kwargs, time_delta))
                        # Execute the original function
                        return f(*args, **kwargs)
                    return wrapper
                
                setattr(self.st_module, func_name, make_wrapper(original_func))
                
    def _unwrap_streamlit_functions(self):
        """Restore original Streamlit functions."""
        for func_name, original_func in self.original_functions.items():
            setattr(self.st_module, func_name, original_func)
            
    def replay(self, speed: float = 1.0) -> None:
        """Replay recorded Streamlit actions.
        
        Args:
            speed: Speed multiplier for replay (1.0 = normal speed)
        """
        for action in self.actions:
            if action.timestamp > 0:
                time.sleep(action.timestamp / speed)
            action.func(*action.args, **action.kwargs)
