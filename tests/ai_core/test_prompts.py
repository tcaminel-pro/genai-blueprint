"""Tests for prompt utilities and wrapper functions."""

from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate

from src.ai_core.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    dedent_ws,
    def_prompt,
    dict_input_message,
    list_input_message,
)


class TestPromptUtils:
    """Test cases for prompt utilities."""

    def test_dedent_ws_basic(self):
        """Test basic dedent functionality."""
        text = "    Hello\n    World"
        result = dedent_ws(text)
        assert result == "Hello\nWorld"

    def test_dedent_ws_with_tabs(self):
        """Test dedent with tabs conversion."""
        text = "\tHello\n\tWorld"
        result = dedent_ws(text)
        assert result == "Hello\nWorld"

    def test_dedent_ws_mixed_whitespace(self):
        """Test dedent with mixed tabs and spaces."""
        text = "    \tHello\n    \tWorld"
        result = dedent_ws(text)
        assert result == "Hello\nWorld"

    def test_dedent_ws_leading_whitespace_only(self):
        """Test dedent removes only leading whitespace."""
        text = "    Hello World  "
        result = dedent_ws(text)
        assert result == "Hello World  "

    def test_def_prompt_basic(self):
        """Test basic prompt creation."""
        prompt = def_prompt(user="Hello")
        assert isinstance(prompt, BasePromptTemplate)
        assert isinstance(prompt, ChatPromptTemplate)

    def test_def_prompt_with_system(self):
        """Test prompt creation with system message."""
        prompt = def_prompt(system="You are helpful", user="Hello")
        messages = prompt.messages
        assert len(messages) == 2

    def test_def_prompt_with_other_messages(self):
        """Test prompt creation with additional messages."""
        prompt = def_prompt(system="You are helpful", user="Hello", other_msg={"placeholder": "{scratchpad}"})
        messages = prompt.messages
        assert len(messages) == 3

    def test_def_prompt_dedent_removes_common_whitespace(self):
        """Test that def_prompt applies dedent_ws to remove common whitespace."""
        system_msg = """
            You are a helpful assistant.
            Always be polite.
        """
        user_msg = """
            Hello, can you help me?
            I have a question.
        """

        prompt = def_prompt(system=system_msg, user=user_msg)

        # Check that common leading whitespace is removed
        system_str = str(prompt.messages[0])
        assert "You are a helpful assistant." in system_str
        assert "Always be polite." in system_str

        user_str = str(prompt.messages[1])
        assert "Hello, can you help me?" in user_str
        assert "I have a question." in user_str

    def test_def_prompt_none_system(self):
        """Test prompt creation with None system message."""
        prompt = def_prompt(system=None, user="Hello")
        messages = prompt.messages
        assert len(messages) == 1

    def test_dict_input_message_basic(self):
        """Test dict input message creation."""
        result = dict_input_message("Hello")
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == ("user", "Hello")

    def test_dict_input_message_with_system(self):
        """Test dict input message with system prompt."""
        result = dict_input_message("Hello", system="You are helpful")
        assert len(result["messages"]) == 2
        assert result["messages"][0] == ("user", "Hello")
        assert result["messages"][1] == ("system", "You are helpful")

    def test_list_input_message_basic(self):
        """Test list input message creation."""
        result = list_input_message("Hello")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_list_input_message_with_system(self):
        """Test list input message with system prompt."""
        result = list_input_message("Hello", system="You are helpful")
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "system"
        assert result[1]["content"] == "You are helpful"

    def test_dedent_ws_handles_empty_string(self):
        """Test dedent_ws with empty string."""
        assert dedent_ws("") == ""

    def test_dedent_ws_single_line(self):
        """Test dedent_ws with single line."""
        assert dedent_ws("    Hello") == "Hello"

    def test_dedent_ws_no_leading_whitespace(self):
        """Test dedent_ws when there's no leading whitespace."""
        text = "Hello\nWorld"
        assert dedent_ws(text) == "Hello\nWorld"

    def test_dedent_ws_preserves_newlines(self):
        """Test dedent_ws preserves newlines and formatting."""
        text = "    Line 1\n\n    Line 2\n    Line 3"
        result = dedent_ws(text)
        assert result == "Line 1\n\nLine 2\nLine 3"

    def test_dedent_ws_only_whitespace_lines(self):
        """Test dedent_ws with lines containing only whitespace."""
        text = "    \n    Hello\n    \n    World\n    "
        result = dedent_ws(text)
        assert result == "\nHello\n\nWorld\n"

    def test_dedent_ws_nested_indentation(self):
        """Test dedent_ws with nested indentation."""
        text = """  
        Optional list of section names to limit the search to specific section in the documents.  
        Allowed section name SHOULD BE in that list: \n 
        - level A
            - level Z"""
        result = dedent_ws(text)
        assert (
            result
            == "\nOptional list of section names to limit the search to specific section in the documents.  \nAllowed section name SHOULD BE in that list: \n\n- level A\n    - level Z"
        )

    def test_dedent_ws_mixed_tabs_spaces_consistency(self):
        """Test dedent_ws with mixed tabs and spaces at same logical level."""
        text = "\tLine with tab\n    Line with spaces\n\tAnother tab"
        result = dedent_ws(text)
        # Should normalize tabs to 4 spaces and then dedent
        assert "Line with tab" in result
        assert "Line with spaces" in result
        assert not result.startswith("\t")
        assert not result.startswith("    ")

    def test_def_prompt_default_system(self):
        """Test the default system prompt constant."""
        assert DEFAULT_SYSTEM_PROMPT == ""
