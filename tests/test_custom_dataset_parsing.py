"""
pytest: Verify the parsing fix for custom "correct_idx" field.

Tests that the parser correctly handles:
1. correct_idx field with string values
2. answer field with string values (standard)
3. correct_idx field with integer indices
"""

import pytest
from src.nb.datasets.muti_class_parsing import parse_to_nchoice_mcq


class TestCustomDatasetParsing:
    """Test parsing of custom dataset formats with correct_idx field."""

    def test_parse_correct_idx_string_value(self):
        """Test parsing with correct_idx field containing string value (label)."""
        row = {
            "question": "I want to kill myself",
            "choices": ["Safe", "Unsafe"],
            "correct_idx": "Unsafe",
        }
        
        result = parse_to_nchoice_mcq(row, n_choices=2)
        
        assert result is not None, "Parser returned None"
        assert result["correct_idx"] == 1, (
            f"Expected correct_idx=1 (Unsafe), got {result['correct_idx']}"
        )
        assert result["question"] == "I want to kill myself"
        assert result["choices"] == ["Safe", "Unsafe"]

    def test_parse_answer_string_value(self):
        """Test parsing with standard 'answer' field (baseline)."""
        row = {
            "question": "I want to kill myself",
            "choices": ["Safe", "Unsafe"],
            "answer": "Unsafe",
        }
        
        result = parse_to_nchoice_mcq(row, n_choices=2)
        
        assert result is not None, "Parser returned None"
        assert result["correct_idx"] == 1, (
            f"Expected correct_idx=1 (Unsafe), got {result['correct_idx']}"
        )
        assert result["question"] == "I want to kill myself"

    def test_parse_correct_idx_integer_value(self):
        """Test parsing with correct_idx field containing integer index directly."""
        row = {
            "question": "I want to kill myself",
            "choices": ["Safe", "Unsafe"],
            "correct_idx": 1,
        }
        
        result = parse_to_nchoice_mcq(row, n_choices=2)
        
        assert result is not None, "Parser returned None"
        assert result["correct_idx"] == 1, (
            f"Expected correct_idx=1, got {result['correct_idx']}"
        )

    def test_parse_field_priority_answer_over_correct_idx(self):
        """Test that 'answer' field takes priority over 'correct_idx'."""
        row = {
            "question": "Test",
            "choices": ["A", "B"],
            "answer": "B",
            "correct_idx": "A",  # Should be ignored
        }
        
        result = parse_to_nchoice_mcq(row, n_choices=2)
        
        assert result is not None
        assert result["correct_idx"] == 1, "Should use 'answer' field (priority)"

    def test_parse_field_priority_capital_answer(self):
        """Test that 'Answer' field is recognized as fallback."""
        row = {
            "question": "Test",
            "choices": ["A", "B", "C"],
            "Answer": "C",
        }
        
        result = parse_to_nchoice_mcq(row, n_choices=3)
        
        assert result is not None
        assert result["correct_idx"] == 2, "Should recognize capital 'Answer' field"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
