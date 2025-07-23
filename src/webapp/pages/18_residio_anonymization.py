"""Streamlit page for data anonymization with Microsoft Presidio.

Demonstrates PII detection, anonymization, and reversible de-anonymization capabilities.
"""
# https://python.langchain.com/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/

from typing import Any

import streamlit as st
from faker import Faker
from langchain_experimental.data_anonymizer import (
    PresidioReversibleAnonymizer,
)
from loguru import logger
from presidio_analyzer import Pattern, PatternRecognizer
from pydantic import BaseModel, ConfigDict
from streamlit import session_state as sss


class AnonymizationDemo(BaseModel):
    """Configuration for Presidio anonymization demo."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    anonymizer: Any = None


st.title("Data Anonymization with Presidio")
st.caption("Protect PII (Personally Identifiable Information) using Microsoft Presidio")

# Initialize session state
if "anon" not in sss:
    with st.spinner("load Spacy Model..."):
        anonymizer = PresidioReversibleAnonymizer()

    # Add custom recognizers for company and project names
    company_recognizer = PatternRecognizer(
        supported_entity="COMPANY",
        patterns=[Pattern(name="company_pattern", regex=r"\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\b", score=0.8)],
        context=["company", "organization", "firm", "enterprise"],
    )

    project_recognizer = PatternRecognizer(
        supported_entity="PROJECT",
        patterns=[
            Pattern(name="project_code_pattern", regex=r"\b[A-Z]+-\d+[A-Z]*\b", score=0.8),
            Pattern(name="project_name_pattern", regex=r"\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+\d*)*\b", score=0.8),
        ],
        context=["project", "initiative", "program"],
    )

    # anonymizer.add_recognizer(company_recognizer)
    # anonymizer.add_recognizer(project_recognizer)

    # Add custom operators for fake data generation
    fake = Faker()
    from presidio_anonymizer.entities import OperatorConfig  # noqa: F401

    # anonymizer.add_operators(
    #     {
    #         "COMPANY": OperatorConfig("custom", {"lambda": lambda _: fake.company()}),
    #         "PROJECT": OperatorConfig("custom", {"lambda": lambda _: fake.bothify(text="PROJ-????", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")}),
    #     }
    # )

    sss.anon = AnonymizationDemo(anonymizer=anonymizer)

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
                    deanon_text = sss.anon.anonymizer.deanonymize(sss.anonymized_text)
                    st.subheader("🔓 De-anonymized Text")
                    st.code(deanon_text, language="text")
                except Exception as e:
                    logger.error(f"De-anonymization failed: {e}")
                    st.error(f"De-anonymization error: {str(e)}")

        with st.expander("Anonymization Mapping"):
            if hasattr(sss.anon.anonymizer, "deanon_mapping"):
                mapping = sss.anon.anonymizer.deanon_mapping
                if mapping:
                    st.json(mapping)
                else:
                    st.info("No PII mappings available")
            else:
                st.info("No reversible mappings available")
