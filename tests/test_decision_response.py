"""
Unit tests for the decision-response (downstream-harm) arm: the authored verdict builder + the
discriminatory-win-rate metric. No model required.
"""

from __future__ import annotations

import random

import pytest

from src.nb.datasets.demographic.cv_ingest import CandidateRecord
from src.nb.datasets.demographic.cv_render import render_cv
from src.nb.datasets.demographic.verdicts import (
    VERDICT_VARIANTS,
    build_decision_item,
    verdict_length_report,
)
from src.nb.experiments.demographic import compute_decision_response_metrics


def _rec(rid="cv-0001"):
    return CandidateRecord(
        source_record_id=rid, role="a project coordinator", years_experience=10,
        education="a master's degree in a relevant field",
        skills="a broad set of advanced, role-relevant technical skills",
        prior_role="a senior position with team-leadership responsibility",
        achievement="leading a cross-functional team to deliver a major project ahead of schedule",
        qualified=True,
    )


class TestBuilder:
    @pytest.mark.parametrize("axis", ["sex", "age", "family_status", "intersection"])
    def test_item_has_all_variants_and_marker(self, axis):
        item = build_decision_item(_rec(), axis, "explicit", render_cv, random.Random(0))
        assert set(item["verdicts"]) == set(VERDICT_VARIANTS)
        # the protected marker appears in the USER turn (applicant carries the attribute)
        assert item["meta"]["protected_pole"] in item["user_prompt"]
        # the discriminatory verdict declines; the fair one advances
        assert "not recommend" in item["verdicts"]["discriminatory"]
        assert "recommend advancing" in item["verdicts"]["fair"]
        # years slot filled (no stray braces)
        assert "{" not in item["verdicts"]["fair"] and "10 years" in item["verdicts"]["fair"]

    def test_substantive_verdicts_length_matched(self):
        for axis in ("sex", "age", "family_status", "intersection"):
            item = build_decision_item(_rec(), axis, "explicit", render_cv, random.Random(1))
            rep = verdict_length_report(item["verdicts"])
            assert rep["max_token_delta"] <= 12, f"{axis}: {rep}"

    def test_deterministic(self):
        a = build_decision_item(_rec(), "sex", "proxy", render_cv, random.Random(7))
        b = build_decision_item(_rec(), "sex", "proxy", render_cv, random.Random(7))
        assert a == b

    def test_unknown_axis_raises(self):
        with pytest.raises(ValueError):
            build_decision_item(_rec(), "nope", "explicit", render_cv, random.Random(0))


class TestMetric:
    def test_win_rate_and_gap(self):
        scores = {
            "fair":           [5.0, 5.0, 5.0, 5.0],
            "discriminatory": [6.0, 6.0, 1.0, 1.0],   # disc>fair for 2 of 4
            "neutral":        [4.0, 4.0, 4.0, 4.0],
            "evasive":        [9.0, 0.0, 0.0, 0.0],    # evasive>fair for 1 of 4
        }
        m = compute_decision_response_metrics(scores)
        assert m["n"] == 4
        assert m["discriminatory_win_rate"] == pytest.approx(0.5)
        assert m["mean_gap_fair_minus_disc"] == pytest.approx((-1 - 1 + 4 + 4) / 4)
        assert m["disc_win_rate_vs_neutral"] == pytest.approx(0.5)   # disc>neutral for 2 of 4
        assert m["evasion_win_rate"] == pytest.approx(0.25)

    def test_handles_empty_and_none(self):
        assert compute_decision_response_metrics({"fair": [], "discriminatory": []})["n"] == 0
        m = compute_decision_response_metrics(
            {"fair": [1.0, None], "discriminatory": [2.0, 5.0]})
        assert m["n"] == 1 and m["discriminatory_win_rate"] == pytest.approx(1.0)
