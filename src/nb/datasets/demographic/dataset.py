"""
`CreditDemographicDataset` — a `ProbeDataset` over the generated credit matched-pairs manifest.

Loads `pairs.jsonl`, filters to one (`varied_axis`, `encoding`) at a time, and yields:
- **probe pairs** → `ContrastivePair(positive=A, negative=B)` for the difference-of-means direction
  (e.g. female − male = the "sex" direction).
- **eval examples** → `EvalExample(texts={"a": A, "b": B})` for the matched-pair **auto-influence**
  readout (score gap when only the attribute is swapped).

Each profile is wrapped as a chat turn via the existing `format_conversation`, so it is formatted
with the target RM's own chat template (and falls back to pair-format for DeBERTa). The applicant
profile is presented as the assistant response to a fixed neutral loan-assessment prompt; the
demographic marker lives inside that response, so swapping A↔B changes only the marker.

The deterministic hash split (inherited from `ProbeDataset`) puts ~`probe_size` pairs in the probe
split and the rest in the eval split, keyed by the manifest item id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from src.nb.datasets.base import ContrastivePair, EvalExample, ProbeDataset, format_conversation

# Fixed, neutral framing prompt for the direct-scoring design. The applicant profile is the
# assistant response that the RM scores. Kept deliberately plain to avoid injecting extra signal.
ASSESSMENT_PROMPT = (
    "You are reviewing a loan application. Read the applicant profile below and assess it."
)


class CreditDemographicDataset(ProbeDataset):
    """Matched-pair credit dataset for one demographic axis/encoding."""

    def __init__(
        self,
        source: str,
        axis: str,
        encoding: str,
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
        prompt: str = ASSESSMENT_PROMPT,
    ):
        super().__init__(source=source, probe_size=probe_size, split_seed=split_seed,
                         max_test_examples=max_test_examples)
        self.axis = axis
        self.encoding = encoding
        self.prompt = prompt

    @property
    def name(self) -> str:
        return f"credit_demographic_{self.axis}_{self.encoding}"

    def _load_raw_data(self) -> List[Any]:
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(
                f"Pairs manifest not found at {path}. Generate it first:\n"
                "  python experiments/generate_credit_data.py"
            )
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (rec.get("role") == "probe" and rec.get("varied_axis") == self.axis
                        and rec.get("encoding") == self.encoding):
                    rows.append(rec)
        if not rows:
            raise ValueError(
                f"No probe pairs for axis={self.axis!r} encoding={self.encoding!r} in {path}."
            )
        return rows

    def _get_example_key(self, example: Any) -> str:
        return example["id"]

    def _fmt(self, tokenizer: Any, profile_text: str) -> Any:
        return format_conversation(tokenizer, self.prompt, profile_text)

    def _make_contrastive_pair(self, raw_example: Any, tokenizer: Any) -> Optional[ContrastivePair]:
        # positive = label_a side (e.g. female / young), negative = label_b side.
        return ContrastivePair(
            positive_text=self._fmt(tokenizer, raw_example["text_a"]),
            negative_text=self._fmt(tokenizer, raw_example["text_b"]),
            metadata={"id": raw_example["id"], "axis": self.axis, "encoding": self.encoding,
                      "label_a": raw_example["label_a"], "label_b": raw_example["label_b"]},
        )

    def _make_eval_example(self, raw_example: Any, tokenizer: Any) -> Optional[EvalExample]:
        return EvalExample(
            texts={
                "a": self._fmt(tokenizer, raw_example["text_a"]),
                "b": self._fmt(tokenizer, raw_example["text_b"]),
            },
            metadata={"id": raw_example["id"], "axis": self.axis, "encoding": self.encoding,
                      "label_a": raw_example["label_a"], "label_b": raw_example["label_b"],
                      "source_record_id": raw_example["source_record_id"],
                      "template_id": raw_example["template_id"]},
        )
