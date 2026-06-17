"""Utilities for parsing and formatting variable-size multi-class MCQ rows.

This module supports tasks where answer labels are class-like strings
(e.g., "Class 1", "Class 2") and the number of classes is configurable.

Constraints:
- Minimum 2 classes
- Maximum 4 classes
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional


MIN_CLASSES = 2
MAX_CLASSES = 4
DEFAULT_CLASS_LABELS = ["Class 1", "Class 2", "Class 3", "Class 4"]
POSITION_LABELS = ["A", "B", "C", "D"]


def stable_rng(seed: int, key: str) -> random.Random:
    """Deterministic RNG from a seed + string key (stable across processes)."""
    h = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    s = int.from_bytes(h[:8], "big")
    return random.Random(s)


def _split_semicolon_list(value: Any) -> List[str]:
    """Split semicolon-separated strings into a list (also accepts lists)."""
    if value is None:
        return []
    if isinstance(value, list):
        out = [str(v).strip() for v in value]
        return [v for v in out if v]
    if isinstance(value, str):
        out = [p.strip() for p in value.split(";")]
        return [p for p in out if p]
    s = str(value).strip()
    return [s] if s else []


def validate_class_labels(class_labels: Optional[List[str]], num_classes: int) -> List[str]:
    """Validate and normalize class labels for multi-class tasks."""
    if num_classes < MIN_CLASSES or num_classes > MAX_CLASSES:
        raise ValueError(
            f"num_classes must be between {MIN_CLASSES} and {MAX_CLASSES}, got {num_classes}"
        )

    if class_labels is None:
        return DEFAULT_CLASS_LABELS[:num_classes]

    labels = [str(x).strip() for x in class_labels]
    if len(labels) != num_classes:
        raise ValueError(
            f"class_labels length must match num_classes ({num_classes}), got {len(labels)}"
        )
    if any(not x for x in labels):
        raise ValueError("class_labels cannot contain empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("class_labels must be unique")
    return labels


def format_multi_class_prompt(question: str, choices: List[str], class_labels: List[str]) -> str:
    """Format a variable-size multi-class prompt (question + labeled options)."""
    if len(choices) != len(class_labels):
        raise ValueError("choices and class_labels must have the same length")
    choices_text = "\n".join([f"{class_labels[i]}. {c}" for i, c in enumerate(choices)])
    return f"Question: {question}\n\n{choices_text}"


def format_multi_class_response(answer_idx: int, choices: List[str], class_labels: List[str]) -> str:
    """Format assistant response for multi-class position tasks."""
    if answer_idx < 0 or answer_idx >= len(choices):
        raise ValueError(f"answer_idx out of range: {answer_idx}")
    if len(choices) != len(class_labels):
        raise ValueError("choices and class_labels must have the same length")
    label = class_labels[answer_idx]
    target = choices[answer_idx]
    return f"The answer is {label}. {target}."


def parse_to_nchoice_mcq(
    row: Dict[str, Any], *, n_choices: int, seed: int = 42
) -> Optional[Dict[str, Any]]:
    """Parse a dataset row into {question, choices[n], correct_idx}.

    Supports the same schemas as the 4-choice parser while reducing to n choices.
    """
    if n_choices < MIN_CLASSES or n_choices > MAX_CLASSES:
        raise ValueError(
            f"n_choices must be between {MIN_CLASSES} and {MAX_CLASSES}, got {n_choices}"
        )

    question = (
        row.get("Question")
        or row.get("question")
        or row.get("prompt")
        or row.get("input")
        or row.get("instruction")
        or row.get("query")
        or ""
    )
    if not question:
        return None

    # PlausibleQA-style schema (original raw)
    if "question" in row and "answer" in row and "candidate_answers" in row:
        question = row["question"]
        correct_answer = row["answer"]
        candidates = row["candidate_answers"]

        incorrect: List[str] = []
        if isinstance(candidates, dict):
            sorted_cands = sorted(
                candidates.items(),
                key=lambda x: x[1].get("plackett_luce", 100) if isinstance(x[1], dict) else 100,
            )
            incorrect = [c[0] for c in sorted_cands if c[0] != correct_answer]
        elif isinstance(candidates, list):
            incorrect = [c for c in candidates if c != correct_answer]

        if len(incorrect) < n_choices - 1:
            return None

        rng = stable_rng(seed, question)
        wrongs = incorrect[: n_choices - 1]
        choices = [correct_answer] + wrongs
        rng.shuffle(choices)
        return {"question": question, "choices": choices, "correct_idx": choices.index(correct_answer)}

    # PlausibleQA-style schema (processed/simplified)
    if "Best Answer" in row and ("Incorrect Answers" in row or "Correct Answers" in row):
        best = str(row.get("Best Answer", "")).strip()
        if not best:
            return None

        incorrect = _split_semicolon_list(row.get("Incorrect Answers", ""))
        incorrect = [x for x in incorrect if x and x != best]
        if len(incorrect) < n_choices - 1:
            return None

        rng = stable_rng(seed, question)
        wrongs = rng.sample(incorrect, n_choices - 1)
        choices = [best] + wrongs
        rng.shuffle(choices)
        return {"question": question, "choices": choices, "correct_idx": choices.index(best)}

    # List-of-choices schema (MMLU-style; may have > n options)
    choices_source = row.get("choices")
    if choices_source is None:
        for alt in ("options", "answers", "responses"):
            if isinstance(row.get(alt), list):
                choices_source = row.get(alt)
                break

    if isinstance(choices_source, list):
        choices_raw = [str(c).strip() for c in choices_source]
        choices_raw = [c for c in choices_raw if c]
        if len(choices_raw) < n_choices:
            return None

        # Support multiple field names for the correct answer:
        # - "answer" or "Answer" (standard)
        # - "correct_idx" (string label or integer index)
        answer = row.get(
            "answer",
            row.get(
                "Answer",
                row.get(
                    "correct_idx",
                    row.get(
                        "label",
                        row.get(
                            "target",
                            row.get(
                                "correct_answer",
                                row.get("gold", row.get("ground_truth", 0)),
                            ),
                        ),
                    ),
                ),
            ),
        )
        correct_idx: Optional[int] = None
        if isinstance(answer, int):
            correct_idx = answer
        elif isinstance(answer, str):
            a = answer.strip()
            if a in POSITION_LABELS:
                correct_idx = POSITION_LABELS.index(a)
            elif a.isdigit():
                correct_idx = int(a)
            elif a in choices_raw:
                correct_idx = choices_raw.index(a)

        if correct_idx is None or correct_idx < 0 or correct_idx >= len(choices_raw):
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

    # GSM8K-MC style (A/B/C/D columns)
    choices = [
        str(row.get("A", row.get("a", ""))).strip(),
        str(row.get("B", row.get("b", ""))).strip(),
        str(row.get("C", row.get("c", ""))).strip(),
        str(row.get("D", row.get("d", ""))).strip(),
    ][:n_choices]
    if not all(choices):
        return None

    answer = row.get(
        "Answer",
        row.get(
            "answer",
            row.get("correct_idx", row.get("label", row.get("target", "A"))),
        ),
    )
    if isinstance(answer, int):
        answer_idx = answer
    elif isinstance(answer, str):
        a_raw = answer.strip()
        a = a_raw.upper()
        if a in POSITION_LABELS:
            answer_idx = POSITION_LABELS.index(a)
        elif a_raw.isdigit():
            answer_idx = int(a_raw)
        else:
            answer_idx = 0
    else:
        answer_idx = 0

    if answer_idx < 0 or answer_idx >= n_choices:
        return None

    return {"question": question, "choices": choices, "correct_idx": answer_idx}
