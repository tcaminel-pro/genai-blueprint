"""Reversible anonymization with Presidio and fuzzy matching support.

Taken from:
- https://python.langchain.com/v0.1/docs/guides/productionization/safety/presidio_data_anonymization/
- https://python.langchain.com/api_reference/experimental/data_anonymizer/langchain_experimental.data_anonymizer.deanonymizer_matching_strategies.combined_exact_fuzzy_matching_strategy.html
"""

from typing import Any, Dict, List

from faker import Faker
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain_experimental.data_anonymizer.deanonymizer_matching_strategies import (
    combined_exact_fuzzy_matching_strategy,
)
from presidio_analyzer import Pattern, PatternRecognizer
from presidio_anonymizer.entities import OperatorConfig
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.utils.spacy_model_mngr import SpaCyModelManager


class CustomizedPresidioAnonymizer(BaseModel):
    """A configurable anonymizer with reversible operations and fuzzy matching.

    Provides a generic interface for PII detection, anonymization, and deanonymization
    with support for custom recognizers and reversible operations using fuzzy matching.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    analyzed_fields: List[str] = Field(
        default_factory=lambda: ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"]
    )

    company_names: List[str] = Field(default_factory=list)
    product_names: List[str] = Field(default_factory=list)
    spacy_model: str = Field(default="en_core_web_sm")  # use large one in production

    faker_seed: int | None = None
    _anonymizer: PresidioReversibleAnonymizer = PrivateAttr()
    _faker: Faker = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the anonymizer after model creation."""
        # Ensure SpaCy model is available and configure the environment
        SpaCyModelManager.setup_spacy_model(self.spacy_model)

        # Initialize the Presidio reversible anonymizer
        self._anonymizer = PresidioReversibleAnonymizer(
            analyzed_fields=self.analyzed_fields, faker_seed=self.faker_seed
        )

        # Initialize faker for custom operators
        self._faker = Faker(locale=["en-US", "fr-FR"])

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
            self._anonymizer.add_recognizer(company_recognizer)

            # Add custom operator for companies
            self._anonymizer.add_operators(
                {
                    "COMPANY": OperatorConfig(
                        "custom",
                        {"lambda": lambda _: self._faker.bothify(text="COMP####")},
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
            self._anonymizer.add_recognizer(product_recognizer)

            # Add custom operator for products
            self._anonymizer.add_operators(
                {
                    "PRODUCT": OperatorConfig(
                        "custom",
                        {"lambda": lambda _: self._faker.bothify(text="PROD####")},
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
        return self._anonymizer.anonymize(text)

    def deanonymize(
        self,
        text: str,
        use_fuzzy_matching: bool = True,
    ) -> str:
        """Deanonymize text by restoring original PII.

        Args:
            text: Anonymized text to restore
            use_fuzzy_matching: Whether to use fuzzy matching for better results

        Returns:
            Text with original PII restored
        """
        if use_fuzzy_matching:
            return self._anonymizer.deanonymize(
                text,
                deanonymizer_matching_strategy=lambda *args, **kwargs: combined_exact_fuzzy_matching_strategy(
                    *args, **kwargs
                ),
            )
        else:
            return self._anonymizer.deanonymize(text)

    def get_mapping(self) -> Dict[str, Any]:
        """Get the anonymization mapping for inspection.

        Returns:
            Dictionary mapping fake values to original PII
        """
        return getattr(self._anonymizer, "deanonymizer_mapping", {})

    @staticmethod
    def check_spacy_model_status(model_name: str) -> dict[str, Any]:
        """Check the status of the SpaCy model.

        Args:
            model_name: Name of the SpaCy model to check

        Returns:
            Dictionary with model status information
        """
        model_path = SpaCyModelManager.get_model_path(model_name)

        return {
            "model_name": model_name,
            "is_installed": SpaCyModelManager.is_model_installed(model_name),
            "model_path": str(model_path),
            "path_exists": model_path.exists(),
        }

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

        self._anonymizer.add_recognizer(recognizer)

        # Add custom operator
        self._anonymizer.add_operators(
            {
                entity_name: OperatorConfig(
                    "custom",
                    {"lambda": lambda _: self._faker.bothify(text=replacement_format)},
                )
            }
        )


if __name__ == "__main__":
    """Quick test of the anonymizer functionality."""
    print("Testing CustomizedPresidioAnonymizer...")

    # Initialize anonymizer with sample data
    anonymizer = CustomizedPresidioAnonymizer(
        company_names=["Acme Corp", "Tech Solutions"], product_names=["WidgetPro", "CloudMaster"], faker_seed=42
    )

    # Test text with PII
    test_text = """
    John Smith works at Acme Corp and uses WidgetPro for development.
    His email is john.smith@email.com and phone is (555) 123-4567.
    He previously worked at Tech Solutions where he used CloudMaster.
    """

    print("\nOriginal text:")
    print(test_text)

    # Anonymize
    anonymized = anonymizer.anonymize(test_text)
    print("\nAnonymized text:")
    print(anonymized)

    # Deanonymize
    deanonymized = anonymizer.deanonymize(anonymized)
    print("\nDeanonymized text:")
    print(deanonymized)

    # Show mapping
    mapping = anonymizer.get_mapping()
    print("\nAnonymization mapping:")
    for fake, real in mapping.items():
        print(f"  {fake} -> {real}")

    print("\nTest completed!")
