"""
Unit tests for the **legacy synthetic** CV-screening pipeline.

As of 2026-08 the `cv` domain loads real biographies (Bias-in-Bios) — see `test_bios_pipeline.py`.
The synthetic generator exercised here is retained solely to reproduce pre-2026-08 hiring results
(the erasure, reasoning and decision-response arms were all run on it), so these tests target
`cv_ingest` / `cv_render` directly and assert nothing about the `cv` domain registry entry.

Cover the offline stages (generate → render → inject → validate → loader) without any model, plus a
regression that the shared `make_pair` default (credit) is unchanged by the CV `content_label` hook.
"""

from __future__ import annotations

import json
import random

import pytest

from src.nb.datasets.demographic.cv_ingest import (
    CandidateRecord,
    ROLES,
    generate_candidates,
)
from src.nb.datasets.demographic.cv_render import CV_TEMPLATES, render_cv
from src.nb.datasets.demographic.markers import make_pair
from src.nb.datasets.demographic.validate import Thresholds, validate_pair


def _fake_record(rid="cv-test", qualified=True) -> CandidateRecord:
    return CandidateRecord(
        source_record_id=rid,
        role="a project coordinator",
        years_experience=10 if qualified else 1,
        education="a master's degree in a relevant field" if qualified else "no formal degree in the field",
        skills="a broad set of advanced, role-relevant technical skills",
        prior_role="a senior position with team-leadership responsibility",
        achievement="leading a cross-functional team to deliver a major project ahead of schedule",
        qualified=qualified,
    )


# --------------------------------------------------------------------------- generate
class TestGenerate:
    def test_deterministic(self):
        a = generate_candidates(50, seed=7)
        b = generate_candidates(50, seed=7)
        assert [r.source_record_id for r in a] == [r.source_record_id for r in b]
        assert [(r.role, r.years_experience, r.qualified) for r in a] == \
               [(r.role, r.years_experience, r.qualified) for r in b]

    def test_sane_split_and_label(self):
        recs = generate_candidates(400, seed=42)
        assert len(recs) == 400
        n_strong = sum(r.qualified for r in recs)
        assert 0.3 < n_strong / len(recs) < 0.7  # roughly balanced
        for r in recs:
            # strong bucket years ∈ [8,15], weak ∈ [0,2] → qualified ⟺ years ≥ 8
            assert r.qualified == (r.years_experience >= 8)
            assert r.role in ROLES


# --------------------------------------------------------------------------- render
class TestRender:
    def test_neutral_profile_has_no_demographics(self):
        import re

        text = render_cv(_fake_record(), "cv_v1")
        low = text.lower()
        for word in ["woman", "women", "female", "male", "man", "men", "she", "he", "his", "her"]:
            assert re.search(rf"\b{word}\b", low) is None, f"neutral CV leaked {word!r}"

    def test_render_is_deterministic(self):
        r = _fake_record()
        assert render_cv(r, "cv_v1") == render_cv(r, "cv_v1")

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            render_cv(_fake_record(), "nope")

    def test_marker_inserted_at_slot(self):
        marked = render_cv(_fake_record(), "cv_v1", marker=" The applicant is a woman.")
        assert "The applicant is a woman." in marked


# --------------------------------------------------------------------------- markers + gate
class TestMarkersAndGate:
    @pytest.mark.parametrize("axis", ["sex", "age", "family_status", "intersection"])
    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    def test_pair_is_single_axis_and_passes_gate(self, axis, enc):
        pair = make_pair(_fake_record(), "cv_v1", axis, enc, random.Random(0),
                         render_fn=render_cv, content_label="cv_content")
        # stripping each clause yields identical remainders → single-axis
        assert pair.text_a.replace(pair.clause_a, "", 1) == pair.text_b.replace(pair.clause_b, "", 1)
        thr = Thresholds(max_char_delta=40 if axis == "intersection" else 12)
        res = validate_pair(pair, thr)
        assert res.ok, f"{axis}/{enc} failed gate: {res.reasons}"

    def test_intersection_cell_records_all_axes(self):
        pair = make_pair(_fake_record(), "cv_v1", "intersection", "explicit", random.Random(0),
                         render_fn=render_cv, content_label="cv_content")
        assert pair.intersectional_cell["sex"] == "female-vs-male"
        assert pair.intersectional_cell["age"] == "30-vs-50"
        assert pair.intersectional_cell["family_status"] == "parental_leave-vs-continuous"
        assert pair.held_fixed == ["cv_content", "template"]
        assert "woman" in pair.text_a and "30-year-old" in pair.text_a
        assert "man" in pair.text_b and "50-year-old" in pair.text_b

    def test_credit_default_held_fixed_unchanged(self):
        # Regression: the shared make_pair must keep credit's default content label.
        from src.nb.datasets.demographic.ingest import GermanCreditRecord
        from src.nb.datasets.demographic.render import render_profile

        rec = GermanCreditRecord(
            source_record_id="german-test", checking="no checking account", duration_months=12,
            credit_history="existing credits paid back duly until now", purpose="a new car",
            credit_amount_dm=2000, savings="less than 100 EUR", employment_since="1 to 3 years",
            installment_rate_pct=3, property="real estate", other_installment_plans="none",
            housing="owned", existing_credits=1, job="a skilled employee or official",
            num_dependents=1, telephone=True, foreign_worker=True, credit_good=True,
            raw_sex="female", raw_marital="single", raw_age_years=29,
        )
        pair = make_pair(rec, "credit_v1", "intersection", "explicit", random.Random(0))
        assert pair.held_fixed == ["financial_content", "template"]
        assert render_profile is not None  # imported for clarity


# --------------------------------------------------------------------------- loader
class TestLoader:
    def _write_jsonl(self, tmp_path):
        from src.nb.datasets.demographic.manifest import pair_to_record

        recs = [_fake_record(f"cv-{i:04d}", qualified=bool(i % 2)) for i in range(40)]
        rows = []
        for i, r in enumerate(recs):
            p = make_pair(r, "cv_v1", "sex", "explicit", random.Random(i),
                          render_fn=render_cv, content_label="cv_content")
            rows.append(pair_to_record(p, f"cv-sex-explicit-cv_v1-{r.source_record_id}",
                                       role="probe", seed=42, domain="cv"))
        path = tmp_path / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in rows))
        return path

    def _fake_tokenizer(self):
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.chat_template = None  # → format_conversation uses pair format (returns a tuple)
        return tok

    def test_loader_yields_pairs_and_evals(self, tmp_path):
        from src.nb.datasets.demographic.cv_dataset import CVDemographicDataset

        ds = CVDemographicDataset(str(self._write_jsonl(tmp_path)), axis="sex",
                                  encoding="explicit", probe_size=20, split_seed=42)
        assert ds.name == "cv_demographic_sex_explicit"
        tok = self._fake_tokenizer()
        probe_pairs = ds.get_probe_pairs(tok)
        evals = ds.get_eval_examples(tok)
        assert len(probe_pairs) > 0 and len(evals) > 0
        assert len(probe_pairs) + len(evals) == 40
        assert probe_pairs[0].positive_text and probe_pairs[0].negative_text
        assert set(evals[0].texts.keys()) == {"a", "b"}
        assert evals[0].metadata["template_id"] == "cv_v1"

    def test_loader_filters_axis(self, tmp_path):
        from src.nb.datasets.demographic.cv_dataset import CVDemographicDataset

        ds = CVDemographicDataset(str(self._write_jsonl(tmp_path)), axis="age",
                                  encoding="explicit", probe_size=20)
        with pytest.raises(ValueError):  # no age rows in this manifest
            ds.get_probe_pairs(self._fake_tokenizer())
