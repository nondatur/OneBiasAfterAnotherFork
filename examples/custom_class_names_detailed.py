"""
Detailed example: Using custom class names in multi-class position bias experiments.

Shows how custom class names flow through:
1. Dataset loading and parsing
2. Probe building
3. Metrics generation
4. Output format
"""

from src.nb.datasets.muti_class_parsing import (
    validate_class_labels,
    format_multi_class_prompt,
    format_multi_class_response,
    parse_to_nchoice_mcq,
)
from src.nb.datasets.position import compute_position_metrics_with_labels


def example_1_custom_binary_labels():
    """Example 1: Binary task with custom labels (True/False)."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Binary (2-class) with True/False labels")
    print("="*70)

    # Validate custom labels
    labels = validate_class_labels(["True", "False"], num_classes=2)
    print(f"Labels: {labels}")

    # Parse an MCQ row down to 2 classes
    row = {
        "question": "The Earth is flat.",
        "A": "Agree",
        "B": "Disagree",
        "C": "Uncertain",
        "D": "No opinion",
        "Answer": "B",
    }
    
    parsed = parse_to_nchoice_mcq(row, n_choices=2, seed=42)
    print(f"\nParsed question: {parsed['question']}")
    print(f"Parsed choices: {parsed['choices']}")
    print(f"Correct index: {parsed['correct_idx']}")

    # Format prompt and response
    prompt = format_multi_class_prompt(parsed["question"], parsed["choices"], labels)
    response = format_multi_class_response(parsed["correct_idx"], parsed["choices"], labels)
    
    print(f"\nFormatted Prompt:\n{prompt}")
    print(f"\nFormatted Response:\n{response}")

    # Simulate metrics
    rewards = {
        "True":  [0.1, 0.8, 0.2, 0.7],  # Model's scores for "True" class
        "False": [0.9, 0.2, 0.8, 0.3],  # Model's scores for "False" class
    }
    correct_positions = [1, 1, 1, 1]  # All correct answers are at position 1 (False)
    
    metrics = compute_position_metrics_with_labels(rewards, correct_positions, labels)
    
    print(f"\nMetrics with custom labels:")
    for key in sorted(metrics.keys()):
        if isinstance(metrics[key], float):
            print(f"  {key}: {metrics[key]:.4f}")
        else:
            print(f"  {key}: {metrics[key]}")


def example_2_custom_ternary_labels():
    """Example 2: Ternary task with custom labels (Agree/Neutral/Disagree)."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Ternary (3-class) with Agree/Neutral/Disagree labels")
    print("="*70)

    labels = validate_class_labels(
        ["Strongly Agree", "Neutral", "Strongly Disagree"],
        num_classes=3
    )
    print(f"Labels: {labels}")

    row = {
        "question": "Artificial Intelligence will be beneficial for humanity.",
        "choices": ["Yes", "Maybe", "No", "I don't know"],
        "answer": 0,  # Correct is first choice
    }
    
    parsed = parse_to_nchoice_mcq(row, n_choices=3, seed=42)
    print(f"\nParsed question: {parsed['question']}")
    print(f"Parsed choices: {parsed['choices']}")
    print(f"Correct index: {parsed['correct_idx']}")

    prompt = format_multi_class_prompt(parsed["question"], parsed["choices"], labels)
    response = format_multi_class_response(parsed["correct_idx"], parsed["choices"], labels)
    
    print(f"\nFormatted Prompt:\n{prompt}")
    print(f"\nFormatted Response:\n{response}")

    # Simulate evaluation with position bias
    rewards = {
        "Strongly Agree":     [0.8, 0.6, 0.5, 0.4],  # Position 0 bias
        "Neutral":            [0.5, 0.5, 0.6, 0.4],
        "Strongly Disagree":  [0.3, 0.4, 0.2, 0.5],
    }
    # Correct answer is at different positions for each example
    correct_positions = [0, 1, 2, 1]
    
    metrics = compute_position_metrics_with_labels(rewards, correct_positions, labels)
    
    print(f"\nMetrics with custom Likert-scale labels:")
    print(f"  Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Max Position Bias: {metrics['max_position_bias']:.1f}%")
    print(f"  Position distribution:")
    for i, label in enumerate(labels):
        pct_key = f"position_{label}_pct"
        acc_key = f"accuracy_when_{label}"
        print(f"    {label}:")
        print(f"      Selected {metrics[pct_key]:.1f}% of the time")
        print(f"      Accuracy when correct: {metrics[acc_key]:.2%}")


def example_3_custom_quaternary_labels():
    """Example 3: Quaternary task with custom labels (A/B/C/D -> descriptive names)."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Quaternary (4-class) with descriptive names")
    print("="*70)

    labels = validate_class_labels(
        ["Definitely Correct", "Probably Correct", "Probably Wrong", "Definitely Wrong"],
        num_classes=4
    )
    print(f"Labels: {labels}")

    row = {
        "question": "What is the largest planet in our solar system?",
        "A": "Jupiter",
        "B": "Saturn",
        "C": "Neptune",
        "D": "Mars",
        "Answer": "A",
    }
    
    parsed = parse_to_nchoice_mcq(row, n_choices=4, seed=42)
    print(f"\nParsed question: {parsed['question']}")
    print(f"Parsed choices: {parsed['choices']}")
    print(f"Correct index: {parsed['correct_idx']}")

    prompt = format_multi_class_prompt(parsed["question"], parsed["choices"], labels)
    response = format_multi_class_response(parsed["correct_idx"], parsed["choices"], labels)
    
    print(f"\nFormatted Prompt:\n{prompt}")
    print(f"\nFormatted Response:\n{response}")

    # Simulate evaluation with strong position bias
    rewards = {
        "Definitely Correct":  [0.9, 0.3, 0.2, 0.1],
        "Probably Correct":    [0.7, 0.6, 0.3, 0.2],
        "Probably Wrong":      [0.5, 0.5, 0.4, 0.3],
        "Definitely Wrong":    [0.1, 0.2, 0.1, 0.05],
    }
    # Balanced across positions
    correct_positions = [0, 1, 2, 3]
    
    metrics = compute_position_metrics_with_labels(rewards, correct_positions, labels)
    
    print(f"\nMetrics with custom confidence-level labels:")
    print(f"  Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Max Position Bias: {metrics['max_position_bias']:.1f}%")
    print(f"\nDetailed breakdown:")
    for i, label in enumerate(labels):
        pct_key = f"position_{label}_pct"
        acc_key = f"accuracy_when_{label}"
        n_key = f"n_correct_at_{label}"
        print(f"  {label}:")
        print(f"    - Selected {metrics[pct_key]:.1f}% of time")
        print(f"    - Accuracy when correct: {metrics[acc_key]:.2%}")
        print(f"    - Number of correct examples: {int(metrics[n_key])}")


def example_4_yaml_config_with_custom_labels():
    """Show what YAML config looks like with custom labels."""
    print("\n" + "="*70)
    print("EXAMPLE 4: YAML Config with custom class labels")
    print("="*70)

    config_examples = {
        "Binary (True/False)": """
name: position_bias_binary_truthfulness
bias_type: position
dataset_class: position_multi_class
extra:
  num_classes: 2
  class_labels: ["True", "False"]
""",
        "Ternary (Likert)": """
name: position_bias_likert_sentiment
bias_type: position
dataset_class: position_multi_class
extra:
  num_classes: 3
  class_labels: ["Positive", "Neutral", "Negative"]
""",
        "Quaternary (Confidence)": """
name: position_bias_confidence_levels
bias_type: position
dataset_class: position_multi_class
extra:
  num_classes: 4
  class_labels:
    - "Very Confident"
    - "Somewhat Confident"
    - "Somewhat Uncertain"
    - "Very Uncertain"
""",
    }

    for task_type, yaml_snippet in config_examples.items():
        print(f"\n{task_type}:")
        print(yaml_snippet)


def example_5_metric_output_comparison():
    """Show how metric keys change with different custom labels."""
    print("\n" + "="*70)
    print("EXAMPLE 5: How metric keys change with custom labels")
    print("="*70)

    # Simulate same rewards with different label sets
    # NOTE: The rewards dict keys MUST match the labels exactly!
    rewards_default = {
        "Class 1": [0.8, 0.2, 0.9],
        "Class 2": [0.2, 0.8, 0.1],
    }
    
    # Test with default labels
    metrics_default = compute_position_metrics_with_labels(
        rewards_default,
        [0, 1, 0],
        ["Class 1", "Class 2"],
    )
    
    # Test with custom labels - use different keys that match the labels
    metrics_custom = compute_position_metrics_with_labels(
        {
            "Yes": [0.8, 0.2, 0.9],
            "No": [0.2, 0.8, 0.1],
        },
        [0, 1, 0],
        ["Yes", "No"],
    )

    print("\nMetric keys with default labels ['Class 1', 'Class 2']:")
    for key in sorted(metrics_default.keys()):
        if "position" in key or "accuracy" in key:
            print(f"  {key}")

    print("\nMetric keys with custom labels ['Yes', 'No']:")
    for key in sorted(metrics_custom.keys()):
        if "position" in key or "accuracy" in key:
            print(f"  {key}")

    print("\nNote: Legacy aliases (position_A_pct, accuracy_when_A, etc.) are also")
    print("generated automatically for backward compatibility with plotting/reporting.")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# CUSTOM CLASS NAMES IN MULTI-CLASS POSITION BIAS EXPERIMENTS")
    print("#" * 70)

    example_1_custom_binary_labels()
    example_2_custom_ternary_labels()
    example_3_custom_quaternary_labels()
    example_4_yaml_config_with_custom_labels()
    example_5_metric_output_comparison()

    print("\n" + "#" * 70)
    print("# KEY TAKEAWAYS")
    print("#" * 70)
    print("""
1. Custom class names are used EVERYWHERE in the pipeline:
   - Prompts: "Option A. Choice text" becomes "Very Agree. Choice text"
   - Responses: "The answer is A." becomes "The answer is Very Agree."
   - Metric keys: "accuracy_when_A" becomes "accuracy_when_Very Agree"

2. Metric keys are dynamically generated from your labels:
   - position_{label}_pct
   - accuracy_when_{label}
   - n_correct_at_{label}

3. Legacy A/B/C/D aliases are ALWAYS generated for backward compatibility:
   - position_A_pct, accuracy_when_A, etc. still appear in the metrics dict
   - This ensures existing plotting/reporting code doesn't break

4. Constraints:
   - Minimum 2 classes, maximum 4
   - Labels must be unique and non-empty
   - Labels must match num_classes exactly

5. Usage: Pass via YAML config extra.class_labels or programmatically
""")
