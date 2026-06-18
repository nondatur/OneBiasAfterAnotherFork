"""
Practical usage guide for multi-class position bias experiments.

Shows three ways to use the new multi-class parsing and dataset:
1. Via YAML config (recommended for running full experiments)
2. Programmatic instantiation (for custom workflows)
3. Direct parsing (if you just need to reformat MCQ data)
"""

from pathlib import Path
from src.nb.experiments.position import run_position_experiment
from src.nb.experiments.base import ExperimentConfig
from src.nb.experiments.position import PositionBiasExperiment
from src.nb.datasets.position import PositionMultiClassDataset
from src.nb.datasets.muti_class_parsing import (
    format_multi_class_prompt,
    format_multi_class_response,
    parse_to_nchoice_mcq,
    validate_class_labels,
)


# ============================================================================
# USAGE 1: Run via YAML config (recommended)
# ============================================================================

def run_via_yaml_config():
    """Run a complete experiment from YAML config file."""
    config_path = Path("experiments/position_multi_class_example.yaml")
    
    # This loads the config and runs the full pipeline:
    # 1. Load model + tokenizer
    # 2. Load dataset with 3-class labels
    # 3. Build position probe
    # 4. Evaluate baseline and debiased performance
    # 5. Save results and plots
    results = run_position_experiment(config_path)
    
    print(f"Accuracy: {results.baseline_metrics['accuracy']:.2%}")
    print(f"Max position bias: {results.baseline_metrics['max_position_bias']:.1f}%")
    return results


# ============================================================================
# USAGE 2: Programmatic instantiation (for custom workflows)
# ============================================================================

def run_programmatic():
    """Instantiate dataset and experiment programmatically."""
    
    # Step 1: Create config
    config = ExperimentConfig(
        name="my_custom_3class_position_experiment",
        bias_type="position",
        model_path="skywork/Skywork-Reward-Llama-3.1-8B",
        dataset_source="guipenedo/gsm8k-mc",
        dataset_class="position_multi_class",
        probe_size=500,
        max_test_examples=500,
        batch_size=8,
        max_length=2048,
        device="cuda",
        extra={
            "dataset_id": "guipenedo/gsm8k-mc",
            "train_split": "train",
            "eval_split": "test",
            "num_classes": 2,  # Use 2 classes (minimum)
            "class_labels": ["Correct", "Incorrect"],
            "clean_with_correctness": True,
        },
    )

    # Step 2: Create experiment
    experiment = PositionBiasExperiment(config)

    # Step 3: Load model and dataset
    experiment.load_model()
    experiment.load_dataset()

    # Step 4: Run evaluation
    results = experiment.evaluate()

    print(f"Dataset: {experiment.dataset.name}")
    print(f"Position labels: {experiment.dataset.position_labels}")
    print(f"Accuracy: {results.baseline_metrics['accuracy']:.2%}")
    
    return results


# ============================================================================
# USAGE 3: Direct parsing (if you just need to reformat MCQ data)
# ============================================================================

def parse_and_format_example():
    """Show how to parse and format MCQ data for multi-class tasks."""
    
    # Example MCQ row from GSM8K-MC or MMLU
    row = {
        "Question": "What is the capital of France?",
        "A": "London",
        "B": "Paris",
        "C": "Berlin",
        "D": "Madrid",
        "Answer": "B",
    }

    # Parse to 2 choices
    parsed_2class = parse_to_nchoice_mcq(row, n_choices=2, seed=42)
    print("\n--- 2-class parsing ---")
    print(f"Question: {parsed_2class['question']}")
    print(f"Choices: {parsed_2class['choices']}")
    print(f"Correct index: {parsed_2class['correct_idx']}")

    # Parse to 3 choices
    parsed_3class = parse_to_nchoice_mcq(row, n_choices=3, seed=42)
    print("\n--- 3-class parsing ---")
    print(f"Question: {parsed_3class['question']}")
    print(f"Choices: {parsed_3class['choices']}")
    print(f"Correct index: {parsed_3class['correct_idx']}")

    # Format with custom labels
    labels_3 = ["Option Alpha", "Option Beta", "Option Gamma"]
    prompt = format_multi_class_prompt(
        parsed_3class["question"],
        parsed_3class["choices"],
        labels_3,
    )
    response = format_multi_class_response(
        parsed_3class["correct_idx"],
        parsed_3class["choices"],
        labels_3,
    )
    
    print("\n--- Formatted for LLM ---")
    print("PROMPT:")
    print(prompt)
    print("\nRESPONSE:")
    print(response)


# ============================================================================
# USAGE 4: Direct dataset instantiation (for exploratory analysis)
# ============================================================================

def explore_dataset_directly():
    """Directly instantiate the multi-class dataset to explore it."""
    
    # Create dataset with 4 custom labels
    dataset = PositionMultiClassDataset(
        source="guipenedo/gsm8k-mc",
        split="train",
        eval_split="test",
        probe_size=100,
        num_classes=4,
        class_labels=["Class 1", "Class 2", "Class 3", "Class 4"],
    )

    print(f"\nDataset: {dataset.name}")
    print(f"Position labels: {dataset.position_labels}")
    print(f"Number of classes: {dataset.num_classes}")

    # Load and inspect raw data
    raw_data = dataset._load_raw_data()
    print(f"\nLoaded {len(raw_data)} examples")
    
    first_example = raw_data[0]
    print(f"\nFirst example:")
    print(f"  Question: {first_example['question']}")
    print(f"  Choices: {first_example['choices']}")
    print(f"  Correct idx: {first_example['correct_idx']}")


# ============================================================================
# USAGE 5: Validate class labels (helper)
# ============================================================================

def validate_labels_example():
    """Show how to validate class labels."""
    
    # Valid: defaults
    labels = validate_class_labels(None, 3)
    print(f"Default 3 labels: {labels}")

    # Valid: custom
    labels = validate_class_labels(["True", "False"], 2)
    print(f"Custom 2 labels: {labels}")

    # Invalid: wrong count
    try:
        labels = validate_class_labels(["Label1", "Label2"], 3)
    except ValueError as e:
        print(f"Error (expected): {e}")

    # Invalid: out of range
    try:
        labels = validate_class_labels(None, 5)
    except ValueError as e:
        print(f"Error (expected): {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-CLASS POSITION BIAS EXPERIMENT USAGE GUIDE")
    print("=" * 70)

    print("\n[1] Validate label examples...")
    validate_labels_example()

    print("\n[2] Parse and format MCQ data...")
    parse_and_format_example()

    print("\n[3] Explore dataset directly...")
    explore_dataset_directly()

    print("\n[4] Run via programmatic instantiation...")
    print("(Skipping full model load — would require actual model download)")

    print("\n[5] Run via YAML config...")
    print("(Skipping full run — would require actual model download)")
    print("To run: python -m src.nb.experiments.position --config experiments/position_multi_class_example.yaml")

    print("\n" + "=" * 70)
    print("Done! Check the code above to see all usage patterns.")
    print("=" * 70)
