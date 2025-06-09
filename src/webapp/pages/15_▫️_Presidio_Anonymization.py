"""Streamlit page for data anonymization with Microsoft Presidio.

Demonstrates PII detection, anonymization, and reversible de-anonymization capabilities.
"""

from typing import Any

import streamlit as st
from langchain_experimental.data_anonymizer import (
    PresidioReversibleAnonymizer,
)
from loguru import logger
from pydantic import BaseModel, ConfigDict


class AnonymizationDemo(BaseModel):
    """Configuration for Presidio anonymization demo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    anonymizer: Any = None


st.title("Data Anonymization with Presidio")
st.caption("Protect PII (Personally Identifiable Information) using Microsoft Presidio")

# Initialize session state
if "anon" not in st.session_state:
    st.session_state.anon = AnonymizationDemo(anonymizer=PresidioReversibleAnonymizer())

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Text")
    input_text = st.text_area(
        "Enter text containing PII:",
        height=300,
        value="John Doe's email is john.doe@example.com and his phone is 555-123-4567. He lives in New York.",
    )

with col2:
    st.subheader("Anonymized Results")

    if st.button("Anonymize Text"):
        with st.spinner("Detecting and anonymizing PII..."):
            try:
                # Anonymize the text
                st.session_state.anonymized_text = st.session_state.anon.anonymizer.anonymize(input_text)
                st.session_state.show_reversible = True
            except Exception as e:
                logger.error(f"Anonymization failed: {e}")
                st.error(f"Anonymization error: {str(e)}")

    if "anonymized_text" in st.session_state:
        st.subheader("🛡️ Anonymized Text")
        st.code(st.session_state.anonymized_text, language="text")

        with st.expander("Reversible Operations", expanded=True):
            if st.button("De-anonymize Text"):
                try:
                    # De-anonymize the text
                    deanon_text = st.session_state.anon.anonymizer.deanonymize(st.session_state.anonymized_text)
                    st.subheader("🔓 De-anonymized Text")
                    st.code(deanon_text, language="text")
                except Exception as e:
                    logger.error(f"De-anonymization failed: {e}")
                    st.error(f"De-anonymization error: {str(e)}")

        with st.expander("Anonymization Mapping"):
            if hasattr(st.session_state.anon.anonymizer, "deanon_mapping"):
                mapping = st.session_state.anon.anonymizer.deanon_mapping
                if mapping:
                    st.json(mapping)
                else:
                    st.info("No PII mappings available")
            else:
                st.info("No reversible mappings available")

# Documentation sidebar
with st.sidebar:
    st.markdown("### About Presidio")
    st.markdown("""
    Microsoft Presidio helps protect sensitive data by:
    - 🔍 Identifying PII in text
    - 🎭 Anonymizing with realistic fake data
    - 🔄 Supporting reversible anonymization
    """)

    st.markdown("### Supported PII Types")
    st.markdown("""
    - Names
    - Email addresses
    - Phone numbers
    - Credit cards
    - Locations
    - IP addresses
    - And [more...](https://microsoft.github.io/presidio/supported_entities/)
    """)
