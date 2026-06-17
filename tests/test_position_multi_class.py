from __future__ import annotations

import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch

from src.nb.datasets.muti_class_parsing import (
    format_multi_class_prompt,
    format_multi_class_response,
    parse_to_nchoice_mcq,
    validate_class_labels,
)
from src.nb.datasets.position import compute_position_metrics_with_labels
from src.nb.experiments.base import ExperimentConfig, ExperimentResults
from src.nb.experiments.position import PositionBiasExperiment
from src.nb.experiments.plotting import create_position_bias_plot


class TestMultiClassParsing:
    def test_validate_class_labels_defaults(self):
        labels = validate_class_labels(None, 3)
        assert labels == ["Class 1", "Class 2", "Class 3"]

    def test_validate_class_labels_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            validate_class_labels(["C1"], 1)
        with pytest.raises(ValueError):
            validate_class_labels(["C1", "C2", "C3", "C4", "C5"], 5)

    def test_validate_class_labels_requires_unique_and_matching_length(self):
        with pytest.raises(ValueError):
            validate_class_labels(["C1", "C2"], 3)
        with pytest.raises(ValueError):
            validate_class_labels(["C1", "C1"], 2)

    def test_format_multi_class_prompt_and_response(self):
        prompt = format_multi_class_prompt(
            "What is 2+2?",
            ["3", "4", "5"],
            ["Class 1", "Class 2", "Class 3"],
        )
        response = format_multi_class_response(
            1,
            ["3", "4", "5"],
            ["Class 1", "Class 2", "Class 3"],
        )
        assert "Class 1. 3" in prompt
        assert "Class 2. 4" in prompt
        assert response == "The answer is Class 2. 4."

    def test_parse_to_nchoice_mcq_from_list_schema(self):
        row = {
            "question": "Largest planet?",
            "choices": ["Earth", "Mars", "Jupiter", "Venus"],
            "answer": 2,
        }
        parsed = parse_to_nchoice_mcq(row, n_choices=3, seed=123)
        assert parsed is not None
        assert len(parsed["choices"]) == 3
        assert 0 <= parsed["correct_idx"] < 3
        assert "Jupiter" in parsed["choices"]

    def test_parse_to_nchoice_mcq_with_correct_idx_field(self):
        """Test parsing with 'correct_idx' field (string label)."""
        row = {
            "question": "Safety classification",
            "choices": ["Safe", "Unsafe"],
            "correct_idx": "Unsafe",
        }
        parsed = parse_to_nchoice_mcq(row, n_choices=2, seed=42)
        assert parsed is not None
        assert parsed["choices"] == ["Safe", "Unsafe"]
        assert parsed["correct_idx"] == 1
        assert parsed["choices"][parsed["correct_idx"]] == "Unsafe"

    def test_parse_to_nchoice_mcq_with_correct_idx_integer(self):
        """Test parsing with 'correct_idx' field (integer index)."""
        row = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5"],
            "correct_idx": 1,
        }
        parsed = parse_to_nchoice_mcq(row, n_choices=3, seed=42)
        assert parsed is not None
        assert parsed["correct_idx"] == 1
        assert parsed["choices"][parsed["correct_idx"]] == "4"

    def test_parse_to_nchoice_mcq_custom_prompt_options_label(self):
        row = {
            "prompt": "Classify this response",
            "options": ["Safe", "Unsafe"],
            "label": "1",
        }
        parsed = parse_to_nchoice_mcq(row, n_choices=2, seed=42)
        assert parsed is not None
        assert parsed["question"] == "Classify this response"
        assert parsed["choices"] == ["Safe", "Unsafe"]
        assert parsed["correct_idx"] == 1

    def test_parse_to_nchoice_mcq_ab_schema_with_correct_idx_digit_string(self):
        row = {
            "question": "Binary decision",
            "A": "Accept",
            "B": "Reject",
            "correct_idx": "1",
        }
        parsed = parse_to_nchoice_mcq(row, n_choices=2, seed=42)
        assert parsed is not None
        assert parsed["choices"] == ["Accept", "Reject"]
        assert parsed["correct_idx"] == 1

    def test_parse_to_nchoice_mcq_from_abcd_schema(self):
        row = {
            "Question": "2+2?",
            "A": "3",
            "B": "4",
            "C": "5",
            "D": "6",
            "Answer": "B",
        }
        parsed = parse_to_nchoice_mcq(row, n_choices=2, seed=42)
        assert parsed is not None
        assert parsed["choices"] == ["3", "4"]
        assert parsed["correct_idx"] == 1


class TestMultiClassMetrics:
    def test_compute_position_metrics_with_three_labels(self):
        labels = ["Class 1", "Class 2", "Class 3"]
        rewards = {
            "Class 1": [0.9, 0.1, 0.2],
            "Class 2": [0.2, 0.8, 0.1],
            "Class 3": [0.1, 0.2, 0.9],
        }
        correct_positions = [0, 1, 2]
        metrics = compute_position_metrics_with_labels(rewards, correct_positions, labels)

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["accuracy_when_Class 1"] == pytest.approx(1.0)
        assert metrics["accuracy_when_Class 2"] == pytest.approx(1.0)
        assert metrics["accuracy_when_Class 3"] == pytest.approx(1.0)
        assert metrics["position_A_pct"] == pytest.approx(100 / 3)
        assert metrics["position_D_pct"] == pytest.approx(0.0)


class TestMultiClassExperimentDispatch:
    def test_create_dataset_position_multi_class(self):
        config = ExperimentConfig(
            name="test_multi_class",
            bias_type="position",
            model_path="dummy/model",
            dataset_source="guipenedo/gsm8k-mc",
            extra={
                "dataset_id": "guipenedo/gsm8k-mc",
                "train_split": "train",
                "eval_split": "test",
                "num_classes": 3,
                "class_labels": ["Class 1", "Class 2", "Class 3"],
            },
            dataset_class="position_multi_class",
        )

        exp = PositionBiasExperiment(config)
        ds = exp._create_dataset()
        assert ds.name == "position_multi_class"
        assert ds.num_classes == 3
        assert ds.position_labels == ["Class 1", "Class 2", "Class 3"]


class TestPositionProbeFallback:
    def test_build_probe_skips_when_position_bucket_empty(self):
        config = ExperimentConfig(
            name="test_probe_skip",
            bias_type="position",
            model_path="dummy/model",
            dataset_source="dummy",
            dataset_class="position_multi_class",
        )

        exp = PositionBiasExperiment(config)
        exp.model = object()      # type: ignore[assignment]
        exp.tokenizer = object()  # type: ignore[assignment]

        class _DummyDataset:
            position_labels = ["Class 1", "Class 2"]

            def get_position_embeddings_texts(self, tokenizer):
                return {0: ["x"], 1: []}

        exp.dataset = _DummyDataset()  # type: ignore[assignment]

        metadata = exp.build_probe()

        assert metadata["probe_skipped"] is True
        assert metadata["n_basis_vectors"] == 0
        assert metadata["n_embeddings_by_pos"]["Class 2"] == 0
        assert exp.probe is None


class TestMultiClassPositionPlotLabels:
    def test_create_position_bias_plot_uses_custom_labels(self, tmp_path):
        baseline = {
            "accuracy": 0.5,
            "accuracy_when_SAFE": 0.4,
            "accuracy_when_UNSAFE": 0.6,
            "n_correct_at_SAFE": 10,
            "n_correct_at_UNSAFE": 10,
        }
        nulled = {
            "accuracy": 0.55,
            "accuracy_when_SAFE": 0.45,
            "accuracy_when_UNSAFE": 0.65,
            "n_correct_at_SAFE": 10,
            "n_correct_at_UNSAFE": 10,
        }

        with patch("matplotlib.pyplot.close", lambda *args, **kwargs: None):
            create_position_bias_plot(
                baseline_metrics=baseline,
                nulled_metrics=nulled,
                output_path=tmp_path / "plot.png",
                title="Test Plot",
                n_examples=20,
                position_labels=["SAFE", "UNSAFE"],
            )
            ax = plt.gca()
            assert ax.get_xlabel() == "Correct Answer"
            assert [tick.get_text() for tick in ax.get_xticklabels()] == ["SAFE", "UNSAFE"]
            plt.close("all")


class TestMultiClassResultsSerialization:
    def test_to_dict_prunes_legacy_aliases_for_multi_class(self):
        config = ExperimentConfig(
            name="serialize_multi_class",
            bias_type="position",
            model_path="dummy/model",
            dataset_source="dummy",
            dataset_class="position_multi_class",
        )

        baseline = {
            "accuracy": 0.5,
            "accuracy_when_SAFE": 0.4,
            "accuracy_when_UNSAFE": 0.6,
            "position_SAFE_pct": 40.0,
            "position_UNSAFE_pct": 60.0,
            "n_correct_at_SAFE": 10,
            "n_correct_at_UNSAFE": 10,
            "accuracy_when_A": 0.4,
            "accuracy_when_B": 0.6,
            "position_A_pct": 40.0,
            "position_B_pct": 60.0,
            "n_correct_at_A": 10,
            "n_correct_at_B": 10,
        }

        nulled = {
            "accuracy": 0.55,
            "accuracy_when_SAFE": 0.45,
            "accuracy_when_UNSAFE": 0.65,
            "position_SAFE_pct": 45.0,
            "position_UNSAFE_pct": 55.0,
            "n_correct_at_SAFE": 10,
            "n_correct_at_UNSAFE": 10,
            "accuracy_when_A": 0.45,
            "position_A_pct": 45.0,
            "n_correct_at_A": 10,
        }

        results = ExperimentResults(
            config=config,
            baseline_metrics=baseline,
            nulled_metrics=nulled,
            probe_metadata={},
            n_probe_examples=100,
            n_eval_examples=100,
        )

        out = results.to_dict()
        assert "accuracy_when_SAFE" in out["baseline"]
        assert "position_SAFE_pct" in out["baseline"]
        assert "n_correct_at_SAFE" in out["baseline"]
        assert "accuracy_when_A" not in out["baseline"]
        assert "position_A_pct" not in out["baseline"]
        assert "n_correct_at_A" not in out["baseline"]
        assert "accuracy_when_A" not in out["nulled"]

    def test_to_dict_keeps_aliases_for_standard_position(self):
        config = ExperimentConfig(
            name="serialize_standard_position",
            bias_type="position",
            model_path="dummy/model",
            dataset_source="dummy",
            dataset_class="position",
        )

        baseline = {
            "accuracy": 0.5,
            "accuracy_when_A": 0.4,
            "position_A_pct": 40.0,
            "n_correct_at_A": 10,
        }

        results = ExperimentResults(
            config=config,
            baseline_metrics=baseline,
            nulled_metrics=None,
            probe_metadata={},
            n_probe_examples=100,
            n_eval_examples=100,
        )

        out = results.to_dict()
        assert "accuracy_when_A" in out["baseline"]
        assert "position_A_pct" in out["baseline"]
        assert "n_correct_at_A" in out["baseline"]
