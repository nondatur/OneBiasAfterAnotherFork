"""
Tests for the shared demographic-domain registry (`domains.py`) and the cross-influence pairing,
which guard the credit→domain-dispatch refactor of the battery / cross-influence / additivity runners.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.nb.datasets.demographic.domains import DOMAINS, get_domain
from src.nb.datasets.demographic.render import TEMPLATES
from src.nb.datasets.demographic.cv_render import CV_TEMPLATES


class TestRegistry:
    def test_known_domains(self):
        assert set(DOMAINS) == {"credit", "cv"}
        with pytest.raises(ValueError):
            get_domain("nope")

    def test_template_ids_match_renderers(self):
        assert get_domain("credit").template_ids == tuple(sorted(TEMPLATES))
        assert get_domain("cv").template_ids == tuple(sorted(CV_TEMPLATES))

    def test_is_strong_reads_right_field(self):
        cv = get_domain("cv")
        assert cv.is_strong(SimpleNamespace(qualified=True)) is True
        assert cv.is_strong(SimpleNamespace(qualified=False)) is False
        credit = get_domain("credit")
        assert credit.is_strong(SimpleNamespace(credit_good=True)) is True
        assert credit.is_strong(SimpleNamespace(credit_good=False)) is False

    def test_load_records_nonempty(self):
        cv_recs = get_domain("cv").load_records()
        assert len(cv_recs) == 1000 and hasattr(cv_recs[0], "qualified")
        credit_recs = get_domain("credit").load_records()  # uses the downloaded raw file
        assert len(credit_recs) == 1000 and hasattr(credit_recs[0], "credit_good")


class TestBuildPairs:
    def test_pairs_strong_with_weak_and_respects_n(self):
        from experiments.run_crossinfluence import _build_pairs

        recs = ([SimpleNamespace(source_record_id=f"s{i}", flag=True) for i in range(5)]
                + [SimpleNamespace(source_record_id=f"w{i}", flag=False) for i in range(3)])
        is_strong = lambda r: r.flag
        pairs = _build_pairs(recs, n_pairs=10, seed=0, is_strong=is_strong,
                             template_ids=("t1", "t2"))
        assert len(pairs) == 3  # min(10, 5 strong, 3 weak)
        for strong, weak, tid in pairs:
            assert is_strong(strong) and not is_strong(weak)
            assert tid in ("t1", "t2")
        assert [p[2] for p in pairs] == ["t1", "t2", "t1"]  # template cycling

    def test_deterministic(self):
        from experiments.run_crossinfluence import _build_pairs

        recs = ([SimpleNamespace(source_record_id=f"s{i}", flag=True) for i in range(6)]
                + [SimpleNamespace(source_record_id=f"w{i}", flag=False) for i in range(6)])
        kw = dict(n_pairs=4, seed=7, is_strong=lambda r: r.flag, template_ids=("t1", "t2"))
        a = _build_pairs(recs, **kw)
        b = _build_pairs(recs, **kw)
        assert [(s.source_record_id, w.source_record_id, t) for s, w, t in a] == \
               [(s.source_record_id, w.source_record_id, t) for s, w, t in b]
