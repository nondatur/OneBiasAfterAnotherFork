# Multi-Class Position Bias Task

Variable-class (2-4) position bias experiments using custom class labels instead of A/B/C/D.

## Quick Start

### 1. Via YAML Config (Recommended for Full Experiments)

```bash
python -m src.nb.experiments.position --config experiments/position_multi_class_example.yaml
```

**Config structure:**
```yaml
name: position_bias_3class_gsm8k
bias_type: position
model_path: skywork/Skywork-Reward-Llama-3.1-8B
dataset_source: guipenedo/gsm8k-mc
dataset_class: position_multi_class

extra:
  num_classes: 3  # Min 2, max 4
  class_labels:
    - "Option A"
    - "Option B"
    - "Option C"
  clean_with_correctness: true
```

### 2. Programmatic (For Custom Workflows)

```python
from src.nb.experiments.base import ExperimentConfig
from src.nb.experiments.position import PositionBiasExperiment

config = ExperimentConfig(
    name="my_experiment",
    bias_type="position",
    model_path="skywork/Skywork-Reward-Llama-3.1-8B",
    dataset_source="guipenedo/gsm8k-mc",
    dataset_class="position_multi_class",
    extra={
        "num_classes": 3,
        "class_labels": ["Option A", "Option B", "Option C"],
    },
)

experiment = PositionBiasExperiment(config)
experiment.load_model()
experiment.load_dataset()
results = experiment.evaluate()
```

### 3. Direct Parsing (If You Just Need Data Reformatting)

```python
from src.nb.datasets.muti_class_parsing import (
    parse_to_nchoice_mcq,
    format_multi_class_prompt,
    format_multi_class_response,
    validate_class_labels,
)

# Validate labels
labels = validate_class_labels(["True", "False"], num_classes=2)

# Parse MCQ row to 2 choices
parsed = parse_to_nchoice_mcq(
    {"question": "...", "choices": [...], "answer": 1},
    n_choices=2,
    seed=42
)

# Format for LLM
prompt = format_multi_class_prompt(
    parsed["question"],
    parsed["choices"],
    labels
)
response = format_multi_class_response(
    parsed["correct_idx"],
    parsed["choices"],
    labels
)
```

### 4. Direct Dataset Instantiation

```python
from src.nb.datasets.position import PositionMultiClassDataset

dataset = PositionMultiClassDataset(
    source="guipenedo/gsm8k-mc",
    split="train",
    eval_split="test",
    probe_size=500,
    num_classes=3,
    class_labels=["A", "B", "C"],
)

# Load and inspect
raw_data = dataset._load_raw_data()
print(f"Loaded {len(raw_data)} examples")

# Get probe pairs for building direction
probe_pairs = dataset.get_probe_pairs(tokenizer)

# Get evaluation examples
eval_examples = dataset.get_eval_examples(tokenizer)
```

## Key Components

| Module | Purpose |
|--------|---------|
| `src/nb/datasets/muti_class_parsing.py` | Parse/format MCQ with variable class labels (2-4) |
| `src/nb/datasets/position.py` | `PositionMultiClassDataset`, `CorrectnessPositionMultiClassDataset` |
| `src/nb/experiments/position.py` | `PositionBiasExperiment` handles `dataset_class="position_multi_class"` |

## Constraints

- **num_classes**: 2–4 (minimum 2, maximum 4)
- **class_labels**: Optional list of strings matching num_classes (defaults to "Class 1", "Class 2", etc.)
- **Probe building**: Learns which position/class the model prefers, independent of content
- **Metrics**: Generates labels for accuracy_when_X, position_X_pct, etc. where X is your label
- **Supported field names**: The parser supports multiple field names for the correct answer:
  - `"answer"` or `"Answer"` (standard)
  - `"correct_idx"` (string label or integer index)
  - See [DATASET_FORMATS.md](../DATASET_FORMATS.md) for complete documentation

## Metrics Output

For a 3-class task with labels `["Option A", "Option B", "Option C"]`:

```python
{
    "accuracy": 0.75,
    "max_position_bias": 8.3,
    "accuracy_when_Option A": 0.80,
    "accuracy_when_Option B": 0.70,
    "accuracy_when_Option C": 0.75,
    "position_Option A_pct": 35.0,
    "position_Option B_pct": 33.0,
    "position_Option C_pct": 32.0,
    # Legacy aliases for compatibility:
    "position_A_pct": 35.0,
    "accuracy_when_A": 0.80,
    ...
}
```

## Backward Compatibility

- Existing A/B/C/D position task remains unchanged
- Metrics include legacy position_A_pct, accuracy_when_A, etc. as aliases to avoid plotting/reporting breakage
- Old experiments unaffected

## Working with Custom Datasets

The parser automatically handles multiple dataset formats:

**Supported field names for correct answer:**
- `"answer"` or `"Answer"` (standard)
- `"correct_idx"` (string label or integer index)

**Example custom dataset:**
```python
from src.nb.datasets.position import PositionMultiClassDataset

# Your dataset has "correct_idx" field with label values
dataset = PositionMultiClassDataset(
    source="your_custom_dataset",
    num_classes=2,
    class_labels=["Safe", "Unsafe"],
)
# Parser automatically recognizes and uses the "correct_idx" field

# If your dataset uses non-standard field names, preprocess it:
from datasets import load_dataset

dataset = load_dataset("your_dataset")
dataset = dataset.rename_column("my_answer_field", "answer")
dataset = dataset.rename_column("my_choices_field", "choices")

position_dataset = PositionMultiClassDataset(source=dataset, num_classes=2)
```

For complete details on supported dataset formats, see [DATASET_FORMATS.md](../DATASET_FORMATS.md).

## Example Run

See [examples/multi_class_usage.py](../examples/multi_class_usage.py) for a complete walk-through including:
- Label validation
- Parsing 2, 3, and 4-choice variants
- Formatting prompts/responses
- Dataset exploration
- Full experiment pipeline
