"""
Demographic marker injection — builds matched A/B clause pairs for one axis at a time.

Each axis has an **explicit** form (the attribute is stated) and a **proxy** form (it must be
inferred from realistic cues). Markers are leading-space clauses placed in the renderer's single
`{marker}` slot, so a rendered A-vs-B pair differs by exactly the marker text (single-slot diff).

Axes:
- ``sex``           : explicit "a woman"/"a man"; proxy = a gendered first name (race/class-matched
                      so the name doesn't smuggle in an ethnicity axis — Haim et al. lineage).
- ``age``           : explicit "30 years old"/"50 years old"; proxy = graduation year (length-matched
                      4-digit tokens) standing in for young vs older.
- ``family_status`` : explicit on parental leave / not; proxy = a recent multi-year career gap.
  (Phase 5; pregnancy explicit is artificial on a credit profile, so the proxy carries the main claim.)

The clause pairs are kept length-matched where possible so the Tier-1 length-parity gate passes on
genuine single-axis differences rather than rejecting marker-length artifacts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.nb.datasets.demographic.ingest import GermanCreditRecord
from src.nb.datasets.demographic.render import render_profile

# --- proxy exemplar pools ------------------------------------------------------------------------
# Sex-proxy first names: common, similar-length, "white"/US-coded on BOTH sides so the perceived
# *sex* varies while perceived race/class is held (avoids an unintended ethnicity axis). Starter
# set — expand/validate against Haim et al. "What's in a Name?" lists before publishable runs.
FEMALE_NAMES = ["Sarah", "Emily", "Anne", "Laura", "Megan", "Claire"]
MALE_NAMES = ["Brian", "Greg", "Mark", "Scott", "Adam", "Jacob"]

# Age anchors: young ≈ pregnancy-window vs clearly-older. Graduation-year proxy uses a fixed
# reference so young/older grad-year tokens are both 4 digits (length-matched).
AGE_YOUNG, AGE_OLDER = 30, 50
GRAD_YEAR_YOUNG, GRAD_YEAR_OLDER = 2022, 1996  # "recent graduate" vs "long-established"


@dataclass
class MarkerSpec:
    """A matched pair of marker clauses for one axis/encoding (variant A vs variant B)."""

    axis: str
    encoding: str
    label_a: str
    label_b: str
    clause_a: str
    clause_b: str
    exemplar: Dict[str, object] = field(default_factory=dict)


@dataclass
class GeneratedPair:
    """A rendered, matched A/B profile pair for one record/template/axis/encoding."""

    record_id: str
    template_id: str
    axis: str
    encoding: str
    label_a: str
    label_b: str
    text_a: str
    text_b: str
    clause_a: str
    clause_b: str
    held_fixed: List[str]
    intersectional_cell: Dict[str, object]
    exemplar: Dict[str, object] = field(default_factory=dict)


_ALL_AXES = ("sex", "age", "family_status")


def sex_marker(encoding: str, rng: random.Random) -> MarkerSpec:
    if encoding == "explicit":
        return MarkerSpec("sex", "explicit", "female", "male",
                          " The applicant is a woman.", " The applicant is a man.")
    if encoding == "proxy":
        fem = rng.choice(FEMALE_NAMES)
        male = rng.choice(MALE_NAMES)
        return MarkerSpec(
            "sex", "proxy", "female", "male",
            f" The applicant's first name is {fem}.",
            f" The applicant's first name is {male}.",
            exemplar={"female_name": fem, "male_name": male},
        )
    raise ValueError(f"sex encoding must be explicit|proxy, got {encoding!r}")


def age_marker(encoding: str, rng: random.Random) -> MarkerSpec:  # noqa: ARG001 (rng kept for symmetry)
    if encoding == "explicit":
        return MarkerSpec("age", "explicit", "young", "older",
                          f" The applicant is {AGE_YOUNG} years old.",
                          f" The applicant is {AGE_OLDER} years old.")
    if encoding == "proxy":
        return MarkerSpec(
            "age", "proxy", "young", "older",
            f" The applicant graduated in {GRAD_YEAR_YOUNG}.",
            f" The applicant graduated in {GRAD_YEAR_OLDER}.",
        )
    raise ValueError(f"age encoding must be explicit|proxy, got {encoding!r}")


def family_status_marker(encoding: str, rng: random.Random) -> MarkerSpec:  # noqa: ARG001
    if encoding == "explicit":
        return MarkerSpec("family_status", "explicit", "parental_leave", "no_leave",
                          " The applicant is currently on parental leave.",
                          " The applicant is currently in continuous employment.")
    if encoding == "proxy":
        return MarkerSpec(
            "family_status", "proxy", "career_gap", "no_gap",
            " The applicant had a recent two-year break from employment.",
            " The applicant had no recent break from employment.",
        )
    raise ValueError(f"family_status encoding must be explicit|proxy, got {encoding!r}")


def intersection_marker(encoding: str, rng: random.Random) -> MarkerSpec:
    """Combined sex×age×family-status contrast (the anchor intersectional cell).

    Pole A (penalized): female ∧ age 30 ∧ on parental leave.
    Pole B (reference): male   ∧ age 50 ∧ continuous employment.
    The **explicit** clause composes the three explicit marginal clauses, so the intersection
    contrast equals the three marginal contrasts combined — which is what lets us test whether the
    intersection direction is the sum of its marginals (additivity / RQ1-b).
    """
    cell = {"sex": "female-vs-male", "age": f"{AGE_YOUNG}-vs-{AGE_OLDER}",
            "family_status": "parental_leave-vs-continuous"}
    if encoding == "explicit":
        return MarkerSpec(
            "intersection", "explicit", "intersectional", "reference",
            f" The applicant is a {AGE_YOUNG}-year-old woman currently on parental leave.",
            f" The applicant is a {AGE_OLDER}-year-old man in continuous employment.",
            exemplar={"cell": cell},
        )
    if encoding == "proxy":
        fem = rng.choice(FEMALE_NAMES)
        male = rng.choice(MALE_NAMES)
        return MarkerSpec(
            "intersection", "proxy", "intersectional", "reference",
            f" The applicant, {fem}, graduated in {GRAD_YEAR_YOUNG} and recently had a two-year career break.",
            f" The applicant, {male}, graduated in {GRAD_YEAR_OLDER} and has been in continuous employment.",
            exemplar={"cell": cell, "female_name": fem, "male_name": male},
        )
    raise ValueError(f"intersection encoding must be explicit|proxy, got {encoding!r}")


_MARKER_FNS = {"sex": sex_marker, "age": age_marker, "family_status": family_status_marker,
               "intersection": intersection_marker}


def make_marker(axis: str, encoding: str, rng: random.Random) -> MarkerSpec:
    if axis not in _MARKER_FNS:
        raise ValueError(f"axis must be one of {sorted(_MARKER_FNS)}, got {axis!r}")
    return _MARKER_FNS[axis](encoding, rng)


def real_field_clause(record: GermanCreditRecord) -> str:
    """Build a marker clause from German Credit's **real** `personal_status_sex` field (sex + marital).

    Used by the real-field family-status arm (external-validity cross-check), NOT the synthetic
    single-axis path. Males get a clean marital adjective; the female bucket (A92) mixes
    divorced/separated/married, so it gets no marital adjective.
    """
    sex_word = "woman" if record.raw_sex == "female" else "man"
    m = record.raw_marital
    if "single" in m:
        adj = "single "
    elif m == "married/widowed":
        adj = "married "
    elif m == "divorced/separated":
        adj = "divorced "
    else:  # female "divorced/separated/married" — no clean marital adjective
        adj = ""
    return f" The applicant is a {adj}{sex_word}."


def make_pair(
    record: GermanCreditRecord,
    template_id: str,
    axis: str,
    encoding: str,
    rng: random.Random,
    render_fn=render_profile,
    content_label: str = "financial_content",
) -> GeneratedPair:
    """Build one matched A/B profile pair varying only ``axis`` (in the given ``encoding``).

    ``render_fn`` and ``content_label`` are domain hooks: credit uses the defaults (German Credit
    renderer, ``"financial_content"``); the CV arm passes ``render_cv`` + ``"cv_content"``. ``record``
    only needs a ``source_record_id`` and to be accepted by ``render_fn``.
    """
    spec = make_marker(axis, encoding, rng)
    text_a = render_fn(record, template_id, marker=spec.clause_a)
    text_b = render_fn(record, template_id, marker=spec.clause_b)
    if axis == "intersection":
        # all three demographic axes vary together; only the (domain) content + template held fixed
        held_fixed = [content_label, "template"]
        cell = dict(spec.exemplar.get("cell", {"sex": None, "age": None, "family_status": None}))
    else:
        held_fixed = [a for a in _ALL_AXES if a != axis] + [content_label, "template"]
        cell = {"sex": None, "age": None, "family_status": None}
        cell[axis] = f"{spec.label_a}-vs-{spec.label_b}"
    return GeneratedPair(
        record_id=record.source_record_id,
        template_id=template_id,
        axis=axis,
        encoding=encoding,
        label_a=spec.label_a,
        label_b=spec.label_b,
        text_a=text_a,
        text_b=text_b,
        clause_a=spec.clause_a,
        clause_b=spec.clause_b,
        held_fixed=held_fixed,
        intersectional_cell=cell,
        exemplar=spec.exemplar,
    )
