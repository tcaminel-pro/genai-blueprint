"""Streamlit page for data anonymization with Microsoft Presidio.

Demonstrates PII detection, anonymization, and reversible de-anonymization capabilities.
"""
# https://python.langchain.com/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/

from pathlib import Path
from typing import Any, List

import streamlit as st
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict
from streamlit import session_state as sss

from src.ai_extra.presidio_anonymizer import CustomizedPresidioAnonymizer
from src.utils.config_mngr import global_config
from src.webapp.ui_components.config_editor import edit_config_dialog


class AnonymizationDemo(BaseModel):
    """Configuration for Presidio anonymization demo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    anonymizer: Any = None


st.title("Data Anonymization with Presidio")
st.caption("Protect PII (Personally Identifiable Information) using Microsoft Presidio")

CONF_YAML_FILE = "config/demos/presidio_anonymization.yaml"

# Configuration options
with st.sidebar:
    st.header("Configuration")
    
    # Add edit button for configuration
    if st.button(":material/edit: Edit Config", help="Edit anonymization configuration"):
        edit_config_dialog(CONF_YAML_FILE)
    
    # Load configuration from YAML
    try:
        config = global_config().merge_with(CONF_YAML_FILE)
        analyzed_fields = config.get_list("presidio_anonymization_config.analyzed_fields")
        company_names = config.get_list("presidio_anonymization_config.company_names")
        product_names = config.get_list("presidio_anonymization_config.product_names")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        st.error("Error loading configuration. Using defaults.")
        analyzed_fields = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "LOCATION"]
        company_names = ["Microsoft", "Google", "Apple", "Amazon", "Meta", "Tesla", "Netflix"]
        product_names = ["Azure", "AWS", "ChatGPT", "LangChain", "Presidio", "iPhone", "Android"]

    # Display current configuration
    st.subheader("Current Configuration")
    st.write(f"**Analyzed Fields:** {', '.join(analyzed_fields)}")
    st.write(f"**Companies:** {', '.join(company_names)}")
    st.write(f"**Products:** {', '.join(product_names)}")
    
    # Fuzzy matching options
    use_fuzzy_matching = st.checkbox("Use fuzzy matching for deanonymization", value=True)
    fuzzy_threshold = st.slider("Fuzzy matching threshold", 0.0, 1.0, 0.8, 0.05)

# Initialize session state
config_hash = hash(str(analyzed_fields) + str(company_names) + str(product_names))
if "anon" not in sss or sss.get("config_hash") != config_hash:
    with st.spinner("Loading Spacy Model and initializing anonymizer..."):
        anonymizer = CustomizedPresidioAnonymizer(
            analyzed_fields=analyzed_fields,
            faker_seed=42,
            company_names=company_names,
            product_names=product_names,
        )

    sss.anon = AnonymizationDemo(anonymizer=anonymizer)
    sss.config_hash = config_hash

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
