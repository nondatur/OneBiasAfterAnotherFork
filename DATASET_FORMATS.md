# Dataset Input Formats

This codebase supports multiple-choice questions (MCQ) and other text formats for evaluating reward model biases. This document describes the input formats accepted by each dataset loader.

## Table of Contents

1. [Standard MCQ Formats (4-choice)](#standard-mcq-formats-4-choice)
2. [Multi-class MCQ Formats (2-4 choice)](#multi-class-mcq-formats-2-4-choice)
3. [Freeform Text Formats](#freeform-text-formats)
4. [Supported Datasets](#supported-datasets)
5. [Field Name Normalization](#field-name-normalization)

---

## Standard MCQ Formats (4-choice)

Used by: `position`, `sycophancy_mcq`, `uncertainty_mcq`, and related datasets.

The parser automatically converts datasets with >4 options to exactly 4 options while preserving the correct answer.

### Format 1: GSM8K-MC (Column-based)

**Used by**: `guipenedo/gsm8k-mc`

**Schema:**
```python
{
    "Question": str,  # or "question"
    "A": str,         # or "a" (lowercase)
    "B": str,         # or "b"
    "C": str,         # or "c"
    "D": str,         # or "d"
    "Answer": str | int,  # or "answer"
                           # str: "A"/"B"/"C"/"D" or choice text
                           # int: index (0-3)
}
```

**Example:**
```json
{
    "Question": "What is the capital of France?",
    "A": "London",
    "B": "Paris",
    "C": "Berlin",
    "D": "Madrid",
    "Answer": "B"
}
```

**Accepted field names (case-insensitive):**
- Question: `"Question"` or `"question"`
- Options: `"A"`, `"B"`, `"C"`, `"D"` (or lowercase `a`, `b`, `c`, `d`)
- Correct answer: `"Answer"` or `"answer"`

---

### Format 2: MMLU-style (List-based)

**Used by**: `cais/mmlu`, `lukaemon/mmlu` and similar datasets

**Schema:**
```python
{
    "question": str,  # or "Question"
    "choices": [str, str, str, ...],  # List of all options (≥4 options OK)
    "answer": int | str,  # int: index, str: choice text or letter
    # or "Answer" (uppercase)
}
```

**Example:**
```json
{
    "question": "Which planet is closest to the sun?",
    "choices": [
        "Venus",
        "Mercury",
        "Earth",
        "Mars"
    ],
    "answer": 1
}
```

**Alternative with string answer:**
```json
{
    "question": "Which planet is closest to the sun?",
    "choices": ["Venus", "Mercury", "Earth", "Mars"],
    "answer": "Mercury"
}
```

**Accepted field names:**
- Question: `"question"` or `"Question"`
- Options: `"choices"` (must be a list)
- Correct answer: `"answer"`, `"Answer"`, or `"correct_idx"` (NEW)
  - Can be: integer index, letter ("A"/"B"/"C"/"D"), or choice text

---

### Format 3: PlausibleQA (Processed)

**Used by**: Custom PlausibleQA-derived datasets

**Schema (Simplified):**
```python
{
    "Question": str,
    "Best Answer": str,
    "Incorrect Answers": str,  # semicolon-separated
}
```

**Example:**
```json
{
    "Question": "What is the largest ocean?",
    "Best Answer": "Pacific",
    "Incorrect Answers": "Atlantic; Indian; Arctic"
}
```

**Notes:**
- Incorrect answers are separated by semicolons (`;`)
- Parser randomly samples 3 incorrect answers to create 4-choice format
- Deterministic sampling (controlled by `seed` parameter)

---

### Format 4: PlausibleQA (Raw)

**Used by**: `allenai/plausibleqa` (original format)

**Schema:**
```python
{
    "question": str,
    "answer": str,
    "candidate_answers": dict | list,
    # If dict: {answer_str: {"plackett_luce": float, ...}, ...}
    # If list: [answer_str, ...]
}
```

**Example with dict:**
```json
{
    "question": "What does DNA stand for?",
    "answer": "Deoxyribonucleic acid",
    "candidate_answers": {
        "Deoxyribonucleic acid": {"plackett_luce": 0.85},
        "Digital nucleic acid": {"plackett_luce": 0.05},
        "Dynamic nerve acid": {"plackett_luce": 0.03}
    }
}
```

**Notes:**
- If candidates are sorted by `plackett_luce`, top-3 most plausible are used
- If list, random 3 are sampled

---

## Multi-class MCQ Formats (2-4 choice)

Used by: `position_multi_class`, `correctness_position_multi_class`

Supports variable numbers of choices (2-4) with custom class labels.

### Schema

```python
{
    "question": str,
    "choices": [str, str, ...],  # 2-4 choices
    "answer": int | str,  # int: index, str: choice text or letter
    # OR "correct_idx": int | str  # NEW: alternative field name
}
```

**Examples:**

**Binary (2-class):**
```json
{
    "question": "Is this statement safe?",
    "choices": ["Safe", "Unsafe"],
    "correct_idx": "Unsafe"
}
```

**Ternary (3-class):**
```json
{
    "question": "How true is this statement?",
    "choices": ["True", "Neutral", "False"],
    "answer": 1
}
```

**Quaternary (4-class):**
```json
{
    "question": "Is the model response good?",
    "choices": ["Excellent", "Good", "Poor", "Bad"],
    "answer": "Good"
}
```

**Accepted field names:**
- Question: `"question"` or `"Question"`
- Options: `"choices"` (list of 2-4 strings)
- Correct answer: `"answer"`, `"Answer"`, or `"correct_idx"`
  - Can be: integer index (0-3), choice text, or letter ("A"/"B"/"C"/"D")

**Configuration:**
```yaml
dataset_class: position_multi_class
extra:
  num_classes: 3  # 2, 3, or 4
  class_labels:   # Optional, defaults to ["Class 1", "Class 2", ...]
    - "True"
    - "Neutral"
    - "False"
```

---

## Freeform Text Formats

Used by: `position_freeform`, `position_freeform_bigbench`

For generative/freeform tasks where there's no predefined multiple-choice format.

### Schema

```python
{
    "input": str,  # Query/prompt
    "output": str,  # Reference answer
}
```

**Example:**
```json
{
    "input": "Translate to French: Hello, world",
    "output": "Bonjour, le monde"
}
```

**Variants (depending on source):**
- BigBench format: may include additional metadata
- Prompt/Completion pairs: `"prompt"` + `"completion"`

---

## Supported Datasets

### Pre-configured Datasets

| Dataset ID | Format | Choices | Biases | Notes |
|---|---|---|---|---|
| `guipenedo/gsm8k-mc` | GSM8K-MC | 4 | position, length | Math word problems with MCQ |
| `cais/mmlu` | MMLU-style | 4 | position, length | General knowledge MCQ |
| `allenai/plausibleqa` | PlausibleQA Raw | 3+ | sycophancy, position | Commonsense reasoning |
| `bigbench` | Freeform + MCQ | Varied | All | Multiple benchmark formats |

### Using with HuggingFace `load_dataset()`

```python
from datasets import load_dataset
from src.nb.datasets.position import PositionMultiClassDataset

# Load dataset
dataset = load_dataset("guipenedo/gsm8k-mc", split="train")

# Use with PositionMultiClassDataset
position_dataset = PositionMultiClassDataset(
    source="guipenedo/gsm8k-mc",
    num_classes=2,
    class_labels=["Correct", "Incorrect"],
)
```

---

## Field Name Normalization

The parser automatically normalizes common field name variations:

### Question Field
- Accepted: `"question"`, `"Question"`, `"prompt"`, `"input"`
- Normalized to: `question` (lowercase)

### Choices Field
- Accepted: `"choices"`, `"options"`, `"A"/"B"/"C"/"D"` columns
- Normalized to: `choices` (list of strings)

### Correct Answer Field
- Accepted: `"answer"`, `"Answer"`, `"correct_idx"`, `"label"`, `"output"`
- Can be:
  - **Integer**: 0-based index into choices list
  - **String letter**: `"A"`, `"B"`, `"C"`, `"D"` (case-insensitive)
  - **String text**: Exact match to choice text

### Priority Order
When multiple field names exist, parser checks in this order:
1. `"answer"`
2. `"Answer"`
3. `"correct_idx"` (NEW)
4. Default to `0` (first choice)

**Example: Parser handling**
```python
from src.nb.datasets.muti_class_parsing import parse_to_nchoice_mcq

# All of these parse correctly:
row1 = {"question": "...", "choices": [...], "answer": 1}
row2 = {"question": "...", "choices": [...], "Answer": "B"}
row3 = {"question": "...", "choices": [...], "correct_idx": "Choice text"}

for row in [row1, row2, row3]:
    result = parse_to_nchoice_mcq(row, n_choices=2)
    print(result)  # All return valid parsed format
```

---

## Handling Custom Datasets

### If Your Dataset Has Non-standard Field Names

**Option 1: Use HuggingFace `.rename_column()`**
```python
dataset = load_dataset("your_dataset")
dataset = dataset.rename_column("correct_label", "answer")
dataset = dataset.rename_column("options", "choices")

position_dataset = PositionMultiClassDataset(source=dataset, num_classes=2)
```

**Option 2: Use `.map()` to Preprocess**
```python
def preprocess(row):
    return {
        "question": row["prompt"],
        "choices": row["options"],
        "answer": row["label_id"],
    }

dataset = load_dataset("your_dataset")
dataset = dataset.map(preprocess)

position_dataset = PositionMultiClassDataset(source=dataset, num_classes=2)
```

**Option 3: Add Support for Your Format**

If your format is common enough, add a handler to the parser:

```python
# In src/nb/datasets/muti_class_parsing.py
def parse_to_nchoice_mcq(...):
    # ... existing checks ...
    
    # Your custom format
    if "custom_question_field" in row:
        question = row["custom_question_field"]
        choices = row["custom_choices_field"]
        correct_idx = parse_custom_answer(row["custom_answer_field"])
        # ... parse and return
```

---

## Validation & Error Handling

The parser silently filters out invalid rows:

### Rows rejected with `None` return:
- Missing or empty question
- Fewer than N required choices
- Invalid answer index/label
- Answer label not found in choices

### Example: Debugging Parse Failures

```python
from src.nb.datasets.muti_class_parsing import parse_to_nchoice_mcq

rows = [
    {"question": "Q1?", "choices": ["A", "B"], "answer": 0},  # ✓ Valid
    {"question": "", "choices": ["A", "B"], "answer": 0},      # ✗ Empty question
    {"question": "Q2?", "choices": ["A"], "answer": 0},        # ✗ Only 1 choice
    {"question": "Q3?", "choices": ["A", "B"], "answer": 5},   # ✗ Invalid index
]

for row in rows:
    result = parse_to_nchoice_mcq(row, n_choices=2)
    if result is None:
        print(f"Failed to parse: {row}")
    else:
        print(f"Success: {result}")
```

---

## Best Practices

1. **Use standard field names** when possible:
   - `"question"` for the prompt
   - `"choices"` for the options list
   - `"answer"` for the correct answer

2. **For custom datasets**, prefer the MMLU-style format (list-based):
   ```json
   {
       "question": "...",
       "choices": ["option1", "option2", "option3", "option4"],
       "answer": 1
   }
   ```

3. **Test your format** with the parser before running experiments:
   ```python
   from src.nb.datasets.muti_class_parsing import parse_to_nchoice_mcq
   
   sample_row = ... # your dataset row
   result = parse_to_nchoice_mcq(sample_row, n_choices=2)
   
   if result is None:
       print("Row failed to parse!")
       # Debug by checking all fields
   ```

4. **For position bias experiments**, ensure your dataset has ≥2 options (≥4 for standard format):
   - 2-class: minimum 2 choices
   - 4-class: minimum 4 choices

5. **Use deterministic seeding** for reproducibility when reducing >N options:
   ```python
   result = parse_to_nchoice_mcq(row, n_choices=3, seed=42)
   # Same seed = same reduction each time
   ```
