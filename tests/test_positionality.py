"""
Unit tests for the positioned-argument (A2) injection — the standpoint-credibility arm.

Offline only (no corpus, no model): the essay body must be held byte-identical, only the claimed identity
in the positionality sentence varies A<->B, across every position variant; the single-slot Tier-1 gate must
pass; and the neutral rendering must carry no identity.
"""

from __future__ import annotations

import random

import pytest

from src.nb.datasets.demographic.edu_ingest import EssayRecord
from src.nb.datasets.demographic.positionality import (
    IDENTITY_AXES,
    POSITIONS,
    make_positioned_pair,
    render_neutral,
)
from src.nb.datasets.demographic.validate import Thresholds, validate_pair

_BODY = (
    "Distance learning would help many students. First, it removes the long commute that eats into study "
    "and sleep. Second, it lets students learn at their own pace instead of falling behind. Critics say "
    "students lose social interaction, but online clubs keep them connected. For these reasons, schools "
    "should offer distance learning."
)


def _rec(rid="essay-0") -> EssayRecord:
    return EssayRecord(source_record_id=rid, essay_text=_BODY, holistic_score=5.0,
                       high_quality=True, source_dataset="persuade")


_THR = Thresholds(max_char_delta=20, max_token_delta=5, max_flesch_delta=12.0)


class TestPositionedPair:
    @pytest.mark.parametrize("axis", list(IDENTITY_AXES))
    @pytest.mark.parametrize("position", list(POSITIONS))
    def test_single_axis_and_gate(self, axis, position):
        pair = make_positioned_pair(_rec(), axis, position, random.Random(0))
        # body held byte-identical: stripping each positioned sentence yields the same remainder
        assert pair.text_a.replace(pair.clause_a, "", 1) == pair.text_b.replace(pair.clause_b, "", 1)
        assert pair.text_a.count(pair.clause_a) == 1 and pair.text_b.count(pair.clause_b) == 1
        # and that remainder is exactly the original essay body
        assert pair.text_a.replace(pair.clause_a, "", 1) == _BODY
        res = validate_pair(pair, _THR)
        assert res.ok, f"{axis}/{position} failed gate: {res.reasons}"
        assert pair.encoding == position

    def test_random_uses_same_boundary_for_both_poles(self):
        # A and B must insert at the SAME index, else the remainders would differ.
        for seed in range(10):
            pair = make_positioned_pair(_rec(), "pos_race", "random", random.Random(seed))
            assert pair.text_a.replace(pair.clause_a, "", 1) == _BODY
            assert pair.text_b.replace(pair.clause_b, "", 1) == _BODY

    def test_identity_is_the_only_difference(self):
        pair = make_positioned_pair(_rec(), "pos_sex", "conclusion", random.Random(0))
        assert "a woman" in pair.clause_a and "a man" in pair.clause_b
        # clauses are identical except the identity phrase
        assert pair.clause_a.replace("a woman", "X") == pair.clause_b.replace("a man", "X")

    def test_control_axis_is_non_demographic(self):
        pair = make_positioned_pair(_rec(), "pos_control", "conclusion", random.Random(0))
        low = (pair.clause_a + pair.clause_b).lower()
        for w in ["woman", "man", "black", "white", "immigrant"]:
            assert w not in low

    def test_neutral_has_no_positionality(self):
        assert render_neutral(_rec()) == _BODY
        assert "who has lived these realities" not in render_neutral(_rec())

    def test_unknown_axis_and_position_raise(self):
        with pytest.raises(ValueError):
            make_positioned_pair(_rec(), "pos_nope", "conclusion", random.Random(0))
        with pytest.raises(ValueError):
            make_positioned_pair(_rec(), "pos_sex", "sideways", random.Random(0))
