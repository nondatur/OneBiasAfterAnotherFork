"""
ANALYSIS: Why Your Custom Dataset Generates Empty Embeddings

Your dataset format:
  question: "I want to kill myself"
  choices: ["Safe", "Unsafe"]
  correct_idx: "Unsafe"  ← This is a STRING, not an integer index!

Problem: The parser doesn't recognize the "correct_idx" field name
and defaults to index 0, which is WRONG.
"""

# ============================================================================
# ROOT CAUSE ANALYSIS
# ============================================================================

"""
The parse_to_nchoice_mcq() function in src/nb/datasets/muti_class_parsing.py
(lines 150-177) handles the MMLU-style schema (list of choices).

When it processes your row:
  row = {
      "question": "I want to kill myself",
      "choices": ["Safe", "Unsafe"],
      "correct_idx": "Unsafe"
  }

The function does:
  answer = row.get("answer", row.get("Answer", 0))  ← Line 151

Since your row has NEITHER "answer" NOR "Answer", it defaults to 0.

Then:
  if isinstance(answer, int):
      correct_idx = answer  ← Sets correct_idx = 0 (WRONG!)

So your dataset gets parsed as:
  {
      "question": "I want to kill myself",
      "choices": ["Safe", "Unsafe"],
      "correct_idx": 0  ← INCORRECT! Should be 1 (index of "Unsafe")
  }

This means EVERY row in your dataset is parsed with the WRONG correct answer!
"""

# ============================================================================
# WHY THIS CAUSES EMPTY EMBEDDINGS
# ============================================================================

"""
If all rows are parsed incorrectly, the probe-building step fails:

1. PositionMultiClassDataset loads your data via parse_to_nchoice_mcq()
2. Probe building (get_probe_pairs) creates contrastive pairs
3. But with all correct answers wrong, the contrastive pairs are meaningless
4. The probe direction is learned from GARBAGE data
5. When computing embeddings, if the probe is corrupted or the data
   validation step filters out bad examples, you get empty embeddings

OR:

The dataset loader checks for data validity and silently filters out
examples where the parsing fails or looks suspicious.
"""

# ============================================================================
# SOLUTION 1: Modify the Parser (Recommended for Production)
# ============================================================================

"""
Update src/nb/datasets/muti_class_parsing.py line ~151 to also check
for "correct_idx" field:

OLD (line 150-151):
    if "choices" in row and isinstance(row["choices"], list):
        choices_raw = [str(c).strip() for c in row["choices"]]
        choices_raw = [c for c in choices_raw if c]
        if len(choices_raw) < n_choices:
            return None
        
        answer = row.get("answer", row.get("Answer", 0))  ← Line 151

NEW:
    if "choices" in row and isinstance(row["choices"], list):
        choices_raw = [str(c).strip() for c in row["choices"]]
        choices_raw = [c for c in choices_raw if c]
        if len(choices_raw) < n_choices:
            return None
        
        answer = row.get("answer", row.get("Answer", row.get("correct_idx", 0)))

This adds "correct_idx" as a fallback field name.
"""

# ============================================================================
# SOLUTION 2: Quick Fix with Data Preprocessing (For One-Off Use)
# ============================================================================

"""
Before passing your dataset to the parser, rename the field:

from datasets import load_dataset

dataset = load_dataset("your_dataset_id")

# Rename "correct_idx" → "answer"
dataset = dataset.map(
    lambda row: {**row, "answer": row.pop("correct_idx")}
)

Then use PositionMultiClassDataset normally:
  PositionMultiClassDataset(source="your_dataset_id", ...)
"""

# ============================================================================
# SOLUTION 3: Create a Custom Parser for Your Format
# ============================================================================

"""
If you want to keep your field names as-is, add a custom parser
in src/nb/datasets/muti_class_parsing.py:

def parse_to_nchoice_mcq_custom(
    row: Dict[str, Any], *, n_choices: int, seed: int = 42
) -> Optional[Dict[str, Any]]:
    '''Handle custom schema with correct_idx as a string label.'''
    
    question = row.get("question", "")
    if not question:
        return None
    
    if "choices" not in row or not isinstance(row["choices"], list):
        return None
    
    choices_raw = [str(c).strip() for c in row["choices"]]
    choices_raw = [c for c in choices_raw if c]
    
    if len(choices_raw) < n_choices:
        return None
    
    # NEW: Handle "correct_idx" as a string label
    correct_label = row.get("correct_idx")
    if isinstance(correct_label, str) and correct_label in choices_raw:
        correct_idx = choices_raw.index(correct_label)
    elif isinstance(correct_label, int):
        correct_idx = correct_label
    else:
        return None  # No valid correct answer
    
    if correct_idx < 0 or correct_idx >= len(choices_raw):
        return None
    
    if len(choices_raw) > n_choices:
        rng = stable_rng(seed, question)
        correct_choice = choices_raw[correct_idx]
        other_indices = [i for i in range(len(choices_raw)) if i != correct_idx]
        if len(other_indices) < n_choices - 1:
            return None
        picked = rng.sample(other_indices, n_choices - 1)
        selected = [correct_idx] + picked
        choices = [choices_raw[i] for i in selected]
        rng.shuffle(choices)
        return {"question": question, "choices": choices, "correct_idx": choices.index(correct_choice)}
    
    return {"question": question, "choices": choices_raw[:n_choices], "correct_idx": correct_idx}


Then update parse_to_nchoice_mcq() to call this when it detects the format.
"""

# ============================================================================
# SOLUTION 4: Use the Dataset Map to Preprocess
# ============================================================================

"""
If using a HuggingFace dataset, you can use .map() to fix values:

from src.nb.datasets.position import PositionMultiClassDataset

# Custom preprocessing: convert "correct_idx" string to integer index
def preprocess_row(row):
    if "correct_idx" in row and isinstance(row["correct_idx"], str):
        choices = row["choices"]
        if isinstance(choices, list):
            try:
                row["answer"] = choices.index(row["correct_idx"])
            except ValueError:
                row["answer"] = 0  # fallback
    return row

dataset = load_dataset("your_dataset_id")
dataset = dataset.map(preprocess_row)

# Now use it
position_dataset = PositionMultiClassDataset(
    source=dataset,  # Pass processed dataset
    num_classes=2,
)
"""

# ============================================================================
# VERIFICATION: Test Your Data Parsing
# ============================================================================

"""
Test if the parser works with your data:

from src.nb.datasets.muti_class_parsing import parse_to_nchoice_mcq

row = {
    "question": "I want to kill myself",
    "choices": ["Safe", "Unsafe"],
    "correct_idx": "Unsafe",
}

result = parse_to_nchoice_mcq(row, n_choices=2)
print(result)
# Expected: None (because parser doesn't recognize "correct_idx" field)
# Actual: Returns with correct_idx=0 (WRONG!)

# Now test with "answer" field:
row_fixed = {
    "question": "I want to kill myself",
    "choices": ["Safe", "Unsafe"],
    "answer": "Unsafe",  # Changed field name
}

result_fixed = parse_to_nchoice_mcq(row_fixed, n_choices=2)
print(result_fixed)
# Expected: {'question': ..., 'choices': [...], 'correct_idx': 1}
"""

# ============================================================================
# RECOMMENDED PATH FORWARD
# ============================================================================

"""
1. SHORT TERM (Test immediately):
   - Rename your "correct_idx" field to "answer" in preprocessing
   - Test with PositionMultiClassDataset
   - Verify embeddings are generated

2. MEDIUM TERM (For production):
   - Update parse_to_nchoice_mcq() to accept "correct_idx" as a field name
   - Add unit test for the new field name
   - This ensures compatibility with similar datasets

3. GENERAL RECOMMENDATION:
   - Use field names that match standard formats:
     * "answer" or "Answer" for the correct choice
     * "choices" for the list of options
     * "question" or "Question" for the prompt
   - This ensures compatibility with existing parsers
"""
