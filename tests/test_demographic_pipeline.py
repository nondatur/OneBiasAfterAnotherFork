"""
Unit tests for the demographic credit data-creation pipeline.

Cover the offline stages (ingest → render → inject → validate → loader → metric) without any model:
- renderer determinism + demographic-neutrality of the baseline,
- marker injection yields a structural single-axis diff,
- Tier-1 gate accepts clean pairs and rejects content drift,
- the CreditDemographicDataset loader yields ContrastivePair/EvalExample,
- auto-influence metric arithmetic.
"""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

import pytest

from src.nb.datasets.demographic.ingest import GermanCreditRecord, load_german_credit
from src.nb.datasets.demographic.render import render_profile, TEMPLATES
from src.nb.datasets.demographic.markers import make_pair
from src.nb.datasets.demographic.validate import validate_pair, validate_pairs, Thresholds


def _fake_record(rid="german-test") -> GermanCreditRecord:
    return GermanCreditRecord(
        source_record_id=rid, checking="no checking account", duration_months=12,
        credit_history="existing credits paid back duly until now", purpose="a new car",
        credit_amount_dm=2000, savings="less than 100 DM", employment_since="1 to 3 years",
        installment_rate_pct=3, property="real estate", other_installment_plans="none",
        housing="owned", existing_credits=1, job="a skilled employee or official",
        num_dependents=1, telephone=True, foreign_worker=True, credit_good=True,
        raw_sex="female", raw_marital="single", raw_age_years=29,
    )


# --------------------------------------------------------------------------- ingest
def test_ingest_loads_and_decodes():
    recs = load_german_credit()  # uses the downloaded raw file
    assert len(recs) == 1000
    r0 = recs[0]
    assert r0.raw_sex in ("male", "female")
    assert isinstance(r0.credit_amount_dm, int) and r0.credit_amount_dm > 0
    assert isinstance(r0.credit_good, bool)


# --------------------------------------------------------------------------- render
class TestRender:
    def test_neutral_profile_has_no_demographics(self):
        import re

        r = _fake_record()  # raw_sex=female, raw_age_years=29
        text = render_profile(r, "credit_v1")
        low = text.lower()
        # the source age must not appear
        assert str(r.raw_age_years) not in text
        # explicit sex words / pronouns must not appear (word-boundary to avoid e.g. "management")
        for word in ["woman", "women", "female", "male", "she", "he"]:
            assert re.search(rf"\b{word}\b", low) is None, f"neutral profile leaked {word!r}"

    def test_render_is_deterministic(self):
        r = _fake_record()
        assert render_profile(r, "credit_v1") == render_profile(r, "credit_v1")

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            render_profile(_fake_record(), "nope")

    def test_marker_inserted_at_slot(self):
        r = _fake_record()
        marked = render_profile(r, "credit_v1", marker=" The applicant is a woman.")
        assert "The applicant is a woman." in marked


# --------------------------------------------------------------------------- markers + gate
class TestMarkersAndGate:
    @pytest.mark.parametrize("axis", ["sex", "age", "family_status", "intersection"])
    @pytest.mark.parametrize("enc", ["explicit", "proxy"])
    def test_pair_is_single_axis_and_passes_gate(self, axis, enc):
        pair = make_pair(_fake_record(), "credit_v1", axis, enc, random.Random(0))
        # stripping each clause yields identical remainders → single-axis
        assert pair.text_a.replace(pair.clause_a, "", 1) == pair.text_b.replace(pair.clause_b, "", 1)
        res = validate_pair(pair)
        assert res.ok, f"{axis}/{enc} failed gate: {res.reasons}"

    def test_intersection_cell_records_all_axes(self):
        pair = make_pair(_fake_record(), "credit_v1", "intersection", "explicit", random.Random(0))
        assert pair.intersectional_cell["sex"] == "female-vs-male"
        assert pair.intersectional_cell["age"] == "30-vs-50"
        assert pair.intersectional_cell["family_status"] == "parental_leave-vs-continuous"
        assert pair.held_fixed == ["financial_content", "template"]
        assert "woman" in pair.text_a and "30-year-old" in pair.text_a
        assert "man" in pair.text_b and "50-year-old" in pair.text_b

    def test_real_field_clause(self):
        import dataclasses
        from src.nb.datasets.demographic.markers import real_field_clause

        base = _fake_record()

        def clause(sex, marital):
            return real_field_clause(dataclasses.replace(base, raw_sex=sex, raw_marital=marital))

        assert clause("male", "single") == " The applicant is a single man."
        assert clause("male", "married/widowed") == " The applicant is a married man."
        assert clause("male", "divorced/separated") == " The applicant is a divorced man."
        # female mixed bucket → no clean marital adjective
        assert clause("female", "divorced/separated/married") == " The applicant is a woman."

    def test_gate_rejects_content_drift(self):
        pair = make_pair(_fake_record(), "credit_v1", "sex", "explicit", random.Random(0))
        # corrupt a financial fact on one side only → not single-axis anymore
        pair.text_b = pair.text_b.replace("2000 EUR", "9999 EUR")
        res = validate_pair(pair)
        assert not res.ok
        assert any("single-axis" in r or "non-marker" in r for r in res.reasons)

    def test_gate_rejects_length_blowup(self):
        pair = make_pair(_fake_record(), "credit_v1", "sex", "explicit", random.Random(0))
        pair.text_a = pair.text_a + " " + ("padding " * 50)
        res = validate_pair(pair, Thresholds(max_char_delta=12, max_token_delta=3))
        assert not res.ok

    def test_validate_pairs_report(self):
        recs = [_fake_record(f"r{i}") for i in range(5)]
        pairs = [make_pair(r, "credit_v1", "sex", "explicit", random.Random(0)) for r in recs]
        passed, failed, report = validate_pairs(pairs)
        assert report["n_passed"] == 5 and report["n_failed"] == 0


# --------------------------------------------------------------------------- loader
class TestLoader:
    def _write_jsonl(self, tmp_path):
        from src.nb.datasets.demographic.markers import make_pair as mp
        from src.nb.datasets.demographic.manifest import pair_to_record

        recs = [_fake_record(f"german-{i:04d}") for i in range(40)]
        rows = []
        for i, r in enumerate(recs):
            p = mp(r, "credit_v1", "sex", "explicit", random.Random(i))
            rows.append(pair_to_record(p, f"credit-sex-explicit-credit_v1-{r.source_record_id}",
                                       role="probe", seed=42))
        path = tmp_path / "pairs.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in rows))
        return path

    def _fake_tokenizer(self):
        tok = MagicMock()
        tok.chat_template = None  # → format_conversation uses pair format (returns a tuple)
        return tok

    def test_loader_yields_pairs_and_evals(self, tmp_path):
        from src.nb.datasets.demographic.dataset import CreditDemographicDataset

        ds = CreditDemographicDataset(str(self._write_jsonl(tmp_path)), axis="sex",
                                      encoding="explicit", probe_size=20, split_seed=42)
        tok = self._fake_tokenizer()
        probe_pairs = ds.get_probe_pairs(tok)
        evals = ds.get_eval_examples(tok)
        assert len(probe_pairs) > 0 and len(evals) > 0
        assert len(probe_pairs) + len(evals) == 40
        cp = probe_pairs[0]
        assert cp.positive_text and cp.negative_text
        assert set(evals[0].texts.keys()) == {"a", "b"}

    def test_loader_filters_axis(self, tmp_path):
        from src.nb.datasets.demographic.dataset import CreditDemographicDataset

        ds = CreditDemographicDataset(str(self._write_jsonl(tmp_path)), axis="age",
                                      encoding="explicit", probe_size=20)
        with pytest.raises(ValueError):  # no age rows in this manifest
            ds.get_probe_pairs(self._fake_tokenizer())


# --------------------------------------------------------------------------- metric
class TestAutoInfluence:
    def test_metric_arithmetic(self):
        from src.nb.experiments.demographic import compute_auto_influence_metrics

        rewards = {"a": [1.0, 2.0, 0.0, 5.0], "b": [0.0, 1.0, 1.0, 1.0]}
        m = compute_auto_influence_metrics(rewards)
        assert m["n_examples"] == 4
        assert m["pref_a_rate"] == pytest.approx(0.75)        # a>b for 3 of 4
        assert m["mean_gap"] == pytest.approx((1 + 1 - 1 + 4) / 4)
        assert m["auto_influence"] == pytest.approx(abs(0.75 - 0.5) * 2)

    def test_metric_handles_none_and_empty(self):
        from src.nb.experiments.demographic import compute_auto_influence_metrics

        assert compute_auto_influence_metrics({"a": [], "b": []})["n_examples"] == 0
        m = compute_auto_influence_metrics({"a": [1.0, None], "b": [0.0, 2.0]})
        assert m["n_examples"] == 1

    def test_cross_influence_metric(self):
        from src.nb.experiments.demographic import compute_cross_influence

        scores = {
            "strong_neutral":   [5.0, 5.0, 5.0, 5.0],
            "weak_neutral":     [1.0, 1.0, 1.0, 1.0],   # strong>weak always → acc_baseline=1.0
            "weak_protected":   [6.0, 6.0, 1.0, 1.0],   # marker flips 2/4 → acc=0.5
            "weak_reference":   [1.0, 1.0, 1.0, 1.0],   # unchanged → acc=1.0
            "strong_protected": [5.0, 5.0, 5.0, 5.0],
        }
        m = compute_cross_influence(scores)
        assert m["acc_baseline"] == pytest.approx(1.0)
        assert m["acc_weak_protected"] == pytest.approx(0.5)
        assert m["cross_influence"] == pytest.approx(0.5)         # reliability harm
        assert m["cross_influence_reference"] == pytest.approx(0.0)
        assert m["baseline_tracks_quality"] is True

    def test_cross_influence_flags_non_tracking(self):
        from src.nb.experiments.demographic import compute_cross_influence

        # RM doesn't prefer the stronger (acc_baseline≈0.5) → flag non-interpretable
        m = compute_cross_influence({
            "strong_neutral": [1.0, 0.0, 1.0, 0.0],
            "weak_neutral":   [0.0, 1.0, 0.0, 1.0],
            "weak_protected": [0.0, 1.0, 0.0, 1.0],
        })
        assert m["acc_baseline"] == pytest.approx(0.5)
        assert m["baseline_tracks_quality"] is False

    def test_per_template_subgroup_split(self):
        from src.nb.datasets.base import EvalExample
        from experiments.run_demographic_battery import _subgroup_auto_influence

        evs = [EvalExample(texts={}, metadata={"template_id": t})
               for t in ["credit_v1", "credit_v2", "credit_v1", "credit_v2"]]
        # v1 examples (idx 0,2): a>b both → AI=1.0; v2 examples (idx 1,3): a<b both → AI=1.0
        base_org = {"a": [2.0, 0.0, 3.0, 0.0], "b": [1.0, 5.0, 1.0, 9.0]}
        out = _subgroup_auto_influence(base_org, evs, key="template_id")
        assert set(out) == {"credit_v1", "credit_v2"}
        assert out["credit_v1"] == pytest.approx(1.0)  # both a>b
        assert out["credit_v2"] == pytest.approx(1.0)  # both a<b (pref_a_rate=0 → AI=1)
