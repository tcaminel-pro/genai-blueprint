"""Reversible anonymization with Presidio and fuzzy matching support."""

from typing import Any, Dict, List, Optional

from faker import Faker
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain_experimental.data_anonymizer.deanonymizer_matching_strategies import (
    combined_exact_fuzzy_matching_strategy,
)
from presidio_analyzer import Pattern, PatternRecognizer
from presidio_anonymizer.entities import OperatorConfig


class CustomizedPresidioAnonymizer:
    """A configurable anonymizer with reversible operations and fuzzy matching.

    Provides a generic interface for PII detection, anonymization, and deanonymization
    with support for custom recognizers and reversible operations using fuzzy matching.
    """

    def __init__(
        self,
        analyzed_fields: Optional[List[str]] = None,
        faker_seed: Optional[int] = 42,
        company_names: Optional[List[str]] = None,
        product_names: Optional[List[str]] = None,
    ):
        """Initialize the anonymizer with configurable options.

        Args:
            analyzed_fields: List of PII field types to detect (e.g., PERSON, EMAIL_ADDRESS)
            faker_seed: Seed for deterministic fake data generation
            company_names: List of company names to anonymize
            product_names: List of product names to anonymize
        """
        self.analyzed_fields = analyzed_fields or [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
        ]
        self.company_names = company_names or []
        self.product_names = product_names or []

        # Initialize the Presidio reversible anonymizer
        self.anonymizer = PresidioReversibleAnonymizer(
            analyzed_fields=self.analyzed_fields,
            faker_seed=faker_seed,
        )

        # Initialize faker for custom operators
        self.fake = Faker(locale=["en-US", "fr-FR"])

        # Add custom recognizers
        self._add_custom_recognizers()

    def _add_custom_recognizers(self) -> None:
        """Add custom recognizers for companies and products."""
        if self.company_names:
            company_pattern = r"(?i)\b(" + "|".join(map(str, self.company_names)) + r")\b"
            company_recognizer = PatternRecognizer(
                supported_entity="COMPANY",
                patterns=[Pattern(name="company_pattern", regex=company_pattern, score=0.9)],
                context=["company", "organization", "firm", "enterprise", "business"],
            )
            self.anonymizer.add_recognizer(company_recognizer)

            # Add custom operator for companies
            self.anonymizer.add_operators(
                {
                    "COMPANY": OperatorConfig(
                        "custom",
                        {"lambda": lambda _: self.fake.bothify(text="COMP####")},
                    )
                }
            )

        if self.product_names:
            product_pattern = r"(?i)\b(" + "|".join(map(str, self.product_names)) + r")\b"
            product_recognizer = PatternRecognizer(
                supported_entity="PRODUCT",
                patterns=[Pattern(name="product_pattern", regex=product_pattern, score=0.9)],
                context=["product", "service", "solution", "platform", "tool"],
            )
            self.anonymizer.add_recognizer(product_recognizer)

            # Add custom operator for products
            self.anonymizer.add_operators(
                {
                    "PRODUCT": OperatorConfig(
                        "custom",
                        {"lambda": lambda _: self.fake.bothify(text="PROD####")},
                    )
                }
            )

    def anonymize(self, text: str) -> str:
        """Anonymize text by replacing PII with fake data.

        Args:
            text: Input text containing PII

        Returns:
            Anonymized text with PII replaced
        """
        return self.anonymizer.anonymize(text)

    def deanonymize(
        self,
        text: str,
        use_fuzzy_matching: bool = True,
        threshold: float = 0.8,
    ) -> str:
        """Deanonymize text by restoring original PII.

        Args:
            text: Anonymized text to restore
            use_fuzzy_matching: Whether to use fuzzy matching for better results
            threshold: Fuzzy matching threshold (0.0-1.0)

        Returns:
            Text with original PII restored
        """
        if use_fuzzy_matching:
            return self.anonymizer.deanonymize(
                text,
                deanonymizer_matching_strategy=lambda *args, **kwargs: combined_exact_fuzzy_matching_strategy(
                    *args, **kwargs
                ),
            )
        else:
            return self.anonymizer.deanonymize(text)

    def get_mapping(self) -> Dict[str, Any]:
        """Get the anonymization mapping for inspection.

        Returns:
            Dictionary mapping fake values to original PII
        """
        return getattr(self.anonymizer, "deanon_mapping", {})

    def add_custom_recognizer(
        self,
        entity_name: str,
        patterns: List[str],
        context_words: List[str],
        replacement_format: str = "####",
    ) -> None:
        """Add a custom recognizer for specific entity types.

        Args:
            entity_name: Name of the entity (e.g., "EMPLOYEE_ID")
            patterns: List of regex patterns to match
            context_words: List of context words for better recognition
            replacement_format: Format for fake replacement values
        """
        combined_pattern = r"(?i)\b(" + "|".join(patterns) + r")\b"

        recognizer = PatternRecognizer(
            supported_entity=entity_name,
            patterns=[Pattern(name=f"{entity_name}_pattern", regex=combined_pattern, score=0.9)],
            context=context_words,
        )

        self.anonymizer.add_recognizer(recognizer)

        # Add custom operator
        self.anonymizer.add_operators(
            {
                entity_name: OperatorConfig(
                    "custom",
                    {"lambda": lambda _: self.fake.bothify(text=replacement_format)},
                )
            }
        )
