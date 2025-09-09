"""Tokenization demo page inspired by tiktokenviewer (https://tiktokenviewer.streamlit.app/)."""

import random
from typing import Tuple

import streamlit as st
import tiktoken as tk
from annotated_text import annotated_text


def return_light_or_dark(rgb_color: Tuple[int, int, int]) -> str:
    """Determine whether to use light or dark text based on background color."""
    r, g, b = rgb_color
    hsp = (0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b)) ** 0.5
    return "#000000" if hsp > 127.5 else "#FFFFFF"


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    h = color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def get_colors(n: int) -> list[str]:
    """Generate n random colors."""
    return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(n)]


def return_tokens(ids: list[int], encoding) -> dict[int, str]:
    """Return tokens from token IDs."""
    return {i: encoding.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids}


def format_whitespace(t: str, include_ws: bool) -> str:
    """Format whitespace characters for display."""
    if include_ws:
        t = t.replace(" ", "␣")
        t = t.replace("\n", "⏎")
    return t


def visualize_tokens(text: str, model: str, include_ws: bool) -> tuple[list, list, int]:
    """Visualize tokens for the given text and model."""
    encoding = tk.encoding_for_model(model)
    ids = encoding.encode(text)

    if not ids:
        return [], [], 0

    tokens = return_tokens(ids, encoding)
    colors = get_colors(len(ids))
    colors_map = dict(zip(ids, colors, strict=False))

    annotated_tokens = []
    annotated_indices = []

    for token_id, token_text in tokens.items():
        color = colors_map[token_id]
        rgb = hex_to_rgb(color)
        text_color = return_light_or_dark(rgb)

        formatted_text = format_whitespace(token_text, include_ws)
        annotated_tokens.append((formatted_text, str(token_id), color, text_color))
        annotated_indices.append((str(token_id), "", color, text_color))

    return annotated_tokens, annotated_indices, len(ids)


# Page configuration
st.set_page_config(
    page_title="Tokenization Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_TEXT = """Hello, world!       
This is a tokenization demo. Ceci est un exemple de Tokenisation.
----------  
 大家好！这是一个标 记 化演示。
 """
# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = EXAMPLE_TEXT

# Get available models
models = ["gpt-2", "gpt-3.5-turbo", "gpt-4"]

# Main UI
st.title("🔍 Tokenization Demo")
st.markdown("Visualize how different GPT models tokenize text")

# Model selection in sidebar
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Select Model", models, index=1)
    include_whitespace = st.toggle("Show whitespace characters", value=True)

# Main content area
st.header("Text Input")
user_text = st.text_area(
    "Enter text to tokenize:",
    value=st.session_state.input_text,
    height=150,
    key="text_input",
)

# Update session state when text changes
if user_text != st.session_state.input_text:
    st.session_state.input_text = user_text

# Visualize tokens when we have text
if st.session_state.input_text:
    tokens, indices, total_tokens = visualize_tokens(st.session_state.input_text, selected_model, include_whitespace)

    st.header("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tokenized Text")
        annotated_text(*tokens)

    with col2:
        st.subheader("Token Indices")
        annotated_text(*indices)

    st.metric("Total Tokens", total_tokens)

    # Display raw tokens for debugging
    with st.expander("Raw Token Details"):
        encoding = tk.encoding_for_model(selected_model)
        ids = encoding.encode(st.session_state.input_text)
        tokens = return_tokens(ids, encoding)

        for token_id, token_text in tokens.items():
            formatted = format_whitespace(token_text, include_whitespace)
            st.text(f"{token_id}: '{formatted}'")
