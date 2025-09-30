"""Tests for the CustomizedPresidioAnonymizer class."""

from src.ai_extra.custom_presidio_anonymizer import CustomizedPresidioAnonymizer


class TestCustomizedPresidioAnonymizer:
    """Test cases for CustomizedPresidioAnonymizer."""

    def test_basic_anonymization(self):
        """Test basic anonymization of PII in text."""
        anonymizer = CustomizedPresidioAnonymizer(faker_seed=42)

        test_text = "John Smith's email is john.smith@email.com"
        anonymized = anonymizer.anonymize(test_text)

        # Should contain fake replacements
        assert "John Smith" not in anonymized
        assert "john.smith@email.com" not in anonymized
        # Just verify that original PII values are not present
        assert all(item not in anonymized for item in ["John Smith", "john.smith@email.com"])

    def test_company_product_anonymization(self):
        """Test anonymization of custom company and product names."""
        anonymizer = CustomizedPresidioAnonymizer(
            company_names=["Acme Corp"], product_names=["WidgetPro"], faker_seed=42
        )

        test_text = "Alice works at Acme Corp using WidgetPro"
        anonymized = anonymizer.anonymize(test_text)

        # Should anonymize company and product names
        assert "Acme Corp" not in anonymized
        assert "WidgetPro" not in anonymized

    def test_reversible_anonymization(self):
        """Test that anonymization is reversible."""
        anonymizer = CustomizedPresidioAnonymizer(faker_seed=42)

        original = "Contact Bob at 555-123-4567 or bob@example.com"
        anonymized = anonymizer.anonymize(original)
        deanonymized = anonymizer.deanonymize(anonymized)

        # Should restore original text
        assert deanonymized == original

    def test_get_mapping(self):
        """Test that mapping is available after anonymization."""
        anonymizer = CustomizedPresidioAnonymizer(faker_seed=42)

        test_text = "Mary Johnson works at TechCorp"
        anonymizer.anonymize(test_text)
        mapping = anonymizer.get_mapping()

        # Should have some mappings
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_spacy_model_status(self):
        """Test checking SpaCy model status."""
        status = CustomizedPresidioAnonymizer.check_spacy_model_status("en_core_web_sm")

        assert isinstance(status, dict)
        assert "model_name" in status
        assert "is_installed" in status
        assert "model_path" in status
        assert "path_exists" in status

    def test_custom_recognizer(self):
        """Test adding custom recognizers."""
        anonymizer = CustomizedPresidioAnonymizer(faker_seed=42)

        # Add custom recognizer for employee IDs
        anonymizer.add_custom_recognizer(
            entity_name="EMPLOYEE_ID",
            patterns=[r"EMP\d{4}"],
            context_words=["employee", "id", "staff"],
            replacement_format="EMP####",
        )

        test_text = "Employee EMP1234 is on leave"
        anonymized = anonymizer.anonymize(test_text)

        # Should anonymize the employee ID
        assert "EMP1234" not in anonymized

    def test_fuzzy_matching_deanonymization(self):
        """Test deanonymization with fuzzy matching enabled."""
        anonymizer = CustomizedPresidioAnonymizer(company_names=["Acme Corp"], faker_seed=42)

        original = "I work at Acme Corp"
        anonymized = anonymizer.anonymize(original)

        # Test with fuzzy matching disabled
        result_exact = anonymizer.deanonymize(anonymized, use_fuzzy_matching=False)
        assert result_exact == original

        # Test with fuzzy matching enabled
        result_fuzzy = anonymizer.deanonymize(anonymized, use_fuzzy_matching=True)
        assert result_fuzzy == original
