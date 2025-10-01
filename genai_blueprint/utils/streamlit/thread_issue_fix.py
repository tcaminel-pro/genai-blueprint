# Hacks to solve threading issue with Streamlit

import inspect
from typing import Callable, TypeVar

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler

from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Taken from https://github.com/streamlit/streamlit/issues/1326
# and https://stackoverflow.com/a/78976474
# Seems to work

T = TypeVar("T")


def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    def decor(fn: Callable[..., T]) -> Callable[..., T]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> T:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamlitCallbackHandler(parent_container=parent_container)

    for name, fn in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if name.startswith("on_"):
            setattr(st_cb, name, decor(fn))

    return st_cb


def get_streamlit_cb_v2(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    fn_return_type = TypeVar("fn_return_type")

    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamlitCallbackHandler(parent_container)

    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith("on_"):
            setattr(st_cb, method_name, add_streamlit_context(method_func))
    return st_cb
