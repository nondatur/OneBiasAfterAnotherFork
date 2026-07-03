"""
Unit tests for the education (grading) demographic pipeline (Direction 1, education arm).

Cover the offline stages (render -> inject -> validate -> loader) on a tiny inline essay fixture
(no corpus download, no model), plus the new education markers (ethnicity name-grid, grade_level) and
a regression that the shared `make_pair` default (credit) is unchanged by the new `subject` hook.
"""

from __future__ import annotations

import json
import random

import pytest

from src.nb.datasets.demographic.edu_ingest import EssayRecord
from src.nb.datasets.demographic.edu_render import EDU_TEMPLATES, render_essay
from src.nb.datasets.demographic.markers import (
    BLACK_FEMALE_NAMES,
    BLACK_MALE_NAMES,
    FEMALE_NAMES,
    MALE_NAMES,
    make_marker,
    make_pair,
)
from src.nb.datasets.demographic.validate import Thresholds, validate_pair

# A neutral, brace-containing essay body — the renderer must copy it verbatim (never .format it).
_ESSAY_BODY = (
    "In this essay I argue that public libraries remain essential. First, they provide free access "
    "to information for everyone regardless of income. Second, they serve as community hubs. The set "
    "{a, b, c} is used here only to check brace-safety. In conclusion, libraries deserve funding."
)


def _fake_record(rid="essay-test", high_quality=True) -> EssayRecord:
    return EssayRecord(
        source_record_id=rid,
        essay_text=_ESSAY_BODY,
        holistic_score=6.0 if high_quality else 1.0,
        high_quality=high_quality,
        source_dataset="persuade",
    )


# --------------------------------------------------------------------------- render
class TestRender:
    def test_body_is_verbatim_and_brace_safe(self):
        text = render_essay(_fake_record(), "edu_v1")
        assert _ESSAY_BODY in text                      # body copied verbatim, braces intact
        assert text.endswith(_ESSAY_BODY)

    def test_neutral_header_has_no_demographics(self):
        # Only the header must be neutral (the essay body is real text, out of our control).
        header = render_essay(_fake_record(), "edu_v1").replace(_ESSAY_BODY, "")
        low = header.lower()
        for word in ["woman", "man", "she", "he", "white", "black", "grade", "name"]:
            assert word not in low, f"neutral header leaked {word!r}"

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            render_essay(_fake_record(), "nope")

    def test_marker_injected_in_header(self):
        marked = render_essay(_fake_record(), "edu_v1", marker=" The student's first name is Jamal.")
        assert "The student's first name is Jamal." in marked
        assert marked.endswith(_ESSAY_BODY)


# --------------------------------------------------------------------------- markers + gate
class TestMarkersAndGate:
    @pytest.mark.parametrize("axis", ["sex", "ethnicity", "grade_level"])
    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    @pytest.mark.parametrize("tid", list(EDU_TEMPLATES))
    def test_pair_is_single_axis_and_passes_gate(self, axis, enc, tid):
        pair = make_pair(_fake_record(), tid, axis, enc, random.Random(0),
                         render_fn=render_essay, content_label="essay_content", subject="student")
        # stripping each clause yields identical remainders → single-axis (body byte-identical)
        assert pair.text_a.replace(pair.clause_a, "", 1) == pair.text_b.replace(pair.clause_b, "", 1)
        assert pair.text_a.count(pair.clause_a) == 1 and pair.text_b.count(pair.clause_b) == 1
        res = validate_pair(pair, Thresholds())
        assert res.ok, f"{axis}/{enc}/{tid} failed gate: {res.reasons}"
        assert "student" in pair.clause_a  # subject noun threaded through

    def test_ethnicity_proxy_holds_sex_fixed(self):
        # The white/black names on the two poles must be the same sex (ethnicity is the only axis).
        for seed in range(20):
            spec = make_marker("ethnicity", "proxy", random.Random(seed), subject="student")
            white, black = spec.exemplar["white_name"], spec.exemplar["black_name"]
            if spec.exemplar["held_sex"] == "female":
                assert white in FEMALE_NAMES and black in BLACK_FEMALE_NAMES
            else:
                assert white in MALE_NAMES and black in BLACK_MALE_NAMES

    def test_sex_proxy_holds_ethnicity_fixed(self):
        # Sex axis uses the "white"-coded pool on both poles, so ethnicity is held.
        spec = make_marker("sex", "proxy", random.Random(1), subject="student")
        assert spec.exemplar["female_name"] in FEMALE_NAMES
        assert spec.exemplar["male_name"] in MALE_NAMES

    def test_grade_level_poles_are_young_vs_older(self):
        spec = make_marker("grade_level", "proxy", random.Random(0), subject="student")
        assert (spec.label_a, spec.label_b) == ("young", "older")

    def test_credit_default_subject_unchanged(self):
        # Regression: the shared make_marker default subject must keep credit clauses byte-identical.
        spec = make_marker("sex", "explicit", random.Random(0))
        assert spec.clause_a == " The applicant is a woman."
        assert spec.clause_b == " The applicant is a man."


# --------------------------------------------------------------------------- loader
class TestLoader:
    def _write_jsonl(self, tmp_path):
        from src.nb.datasets.demographic.manifest import pair_to_record

        rows = []
        for i in range(40):
            rec = _fake_record(f"essay-{i:04d}", high_quality=bool(i % 2))
            p = make_pair(rec, "edu_v1", "ethnicity", "proxy", random.Random(i),
                          render_fn=render_essay, content_label="essay_content", subject="student")
            rows.append(pair_to_record(p, f"edu-ethnicity-proxy-edu_v1-{rec.source_record_id}",
                                       role="probe", seed=42, domain="education"))
        path = tmp_path / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in rows))
        return path

    def _fake_tokenizer(self):
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.chat_template = None  # → format_conversation uses pair format
        return tok

    def test_loader_yields_pairs_and_evals(self, tmp_path):
        from src.nb.datasets.demographic.edu_dataset import EducationDemographicDataset

        ds = EducationDemographicDataset(str(self._write_jsonl(tmp_path)), axis="ethnicity",
                                         encoding="proxy", probe_size=20, split_seed=42)
        assert ds.name == "education_demographic_ethnicity_proxy"
        tok = self._fake_tokenizer()
        probe_pairs = ds.get_probe_pairs(tok)
        evals = ds.get_eval_examples(tok)
        assert len(probe_pairs) > 0 and len(evals) > 0
        assert len(probe_pairs) + len(evals) == 40
        assert set(evals[0].texts.keys()) == {"a", "b"}
        assert evals[0].metadata["template_id"] == "edu_v1"
