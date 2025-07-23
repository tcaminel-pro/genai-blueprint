"""Streamlit page for data anonymization with Microsoft Presidio.

Demonstrates PII detection, anonymization, and reversible de-anonymization capabilities.
"""
# https://python.langchain.com/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/

from typing import Any

import streamlit as st
from loguru import logger
from pydantic import BaseModel, ConfigDict
from streamlit import session_state as sss

from src.ai_extra.presidio_anonymizer import CustomizedPresidioAnonymizer


class AnonymizationDemo(BaseModel):
    """Configuration for Presidio anonymization demo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    anonymizer: Any = None


st.title("Data Anonymization with Presidio")
st.caption("Protect PII (Personally Identifiable Information) using Microsoft Presidio")

# Configuration options
with st.sidebar:
    st.header("Configuration")

    # Company names configuration
    company_names_input = st.text_input(
        "Company names to anonymize (comma-separated):", value="Microsoft, Google, Apple, Amazon, Meta, Tesla, Netflix"
    )
    company_names = [name.strip() for name in company_names_input.split(",") if name.strip()]

    # Product names configuration
    product_names_input = st.text_input(
        "Product names to anonymize (comma-separated):",
        value="Azure, AWS, ChatGPT, LangChain, Presidio, iPhone, Android",
    )
    product_names = [name.strip() for name in product_names_input.split(",") if name.strip()]

    # Fuzzy matching options
    use_fuzzy_matching = st.checkbox("Use fuzzy matching for deanonymization", value=True)
    fuzzy_threshold = st.slider("Fuzzy matching threshold", 0.0, 1.0, 0.8, 0.05)

# Initialize session state
if "anon" not in sss or sss.get("company_names") != company_names or sss.get("product_names") != product_names:
    with st.spinner("Loading Spacy Model and initializing anonymizer..."):
        anonymizer = CustomizedPresidioAnonymizer(
            analyzed_fields=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "LOCATION"],
            faker_seed=42,
            company_names=company_names,
            product_names=product_names,
        )

    sss.anon = AnonymizationDemo(anonymizer=anonymizer)
    sss.company_names = company_names
    sss.product_names = product_names

# Sample texts for demonstration
sample_texts = {
    "Basic PII": "John Doe's email is john.doe@example.com and his phone is 555-123-4567. He lives in New York.",
    "Company & Product": "Alice Johnson works at Microsoft and uses Azure services. Bob Smith from Google prefers AWS.",
    "Mixed Content": "Contact Sarah Wilson at sarah.wilson@company.com. She works at Tesla developing ChatGPT integrations.",
    "Custom Entities": "Employee EMP-1234 from Capgemini reported an issue with the iPhone app.",
}

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Text")

    # Sample selector
    selected_sample = st.selectbox("Choose a sample:", list(sample_texts.keys()))
    input_text = st.text_area(
        "Enter text containing PII:",
        height=300,
        value=sample_texts[selected_sample],
    )

    # Custom recognizer section
    with st.expander("Add Custom Recognizer"):
        entity_name = st.text_input("Entity Name:", placeholder="e.g., EMPLOYEE_ID")
        patterns = st.text_area("Patterns (regex, one per line):", placeholder=r"EMP-\d{4}\nE\d{6}")
        context_words = st.text_input("Context words (comma-separated):", placeholder="employee, staff, worker")

        if st.button("Add Custom Recognizer"):
            if entity_name and patterns:
                pattern_list = [p.strip() for p in patterns.split("\n") if p.strip()]
                context_list = [c.strip() for c in context_words.split(",") if c.strip()]

                try:
                    sss.anon.anonymizer.add_custom_recognizer(
                        entity_name=entity_name,
                        patterns=pattern_list,
                        context_words=context_list,
                        replacement_format=f"{entity_name[:3].upper()}####",
                    )
                    st.success(f"Added custom recognizer for {entity_name}")
                except Exception as e:
                    st.error(f"Failed to add recognizer: {str(e)}")

with col2:
    st.subheader("Anonymized Results")

    if st.button("Anonymize Text"):
        with st.spinner("Detecting and anonymizing PII..."):
            try:
                # Anonymize the text
                sss.anonymized_text = sss.anon.anonymizer.anonymize(input_text)
                sss.show_reversible = True
            except Exception as e:
                logger.exception(f"Anonymization failed: {e}")
                st.error(f"Anonymization error: {str(e)}")

    if "anonymized_text" in sss:
        st.subheader("🛡️ Anonymized Text")
        st.code(sss.anonymized_text, language="text")

        with st.expander("Reversible Operations", expanded=True):
            if st.button("De-anonymize Text"):
                try:
                    # De-anonymize the text
                    deanon_text = sss.anon.anonymizer.deanonymize(
                        sss.anonymized_text,
                        use_fuzzy_matching=use_fuzzy_matching,
                        threshold=fuzzy_threshold,
                    )
                    st.subheader("🔓 De-anonymized Text")
                    st.code(deanon_text, language="text")
                except Exception as e:
                    logger.error(f"De-anonymization failed: {e}")
                    st.error(f"De-anonymization error: {str(e)}")

        with st.expander("Anonymization Mapping"):
            mapping = sss.anon.anonymizer.get_mapping()
            if mapping:
                st.json(mapping)
            else:
                st.info("No reversible mappings available")
