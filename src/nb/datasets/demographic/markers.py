"""
Demographic marker injection — builds matched A/B clause pairs for one axis at a time.

Each axis has an **explicit** form (the attribute is stated) and a **proxy** form (it must be
inferred from realistic cues). Markers are leading-space clauses placed in the renderer's single
`{marker}` slot, so a rendered A-vs-B pair differs by exactly the marker text (single-slot diff).

Axes:
- ``sex``           : explicit "a woman"/"a man"; proxy = a gendered first name (ethnicity-matched
                      "white"-coded on both sides so the name doesn't smuggle in an ethnicity axis).
- ``age``           : explicit "30 years old"/"50 years old"; proxy = graduation year (length-matched
                      4-digit tokens) standing in for young vs older.
- ``family_status`` : explicit on parental leave / not; proxy = a recent multi-year career gap.
- ``ethnicity``     : (education arm) proxy = a first name coded to different origins holding sex fixed
                      (Haim / Bertrand-Mullainathan name lists); explicit = a stated race. A name-proxy
                      effect, not pure ethnicity (names conflate ethnicity with class/region).
- ``grade_level``   : (education arm) an age/education-stage proxy — school pupil vs final-year
                      university student; explicit = a stated student age.

The ``subject`` noun ("applicant" by default; "student" for the education arm) is threaded through so
the clauses read naturally per domain; the default keeps the credit/CV clauses byte-identical.

The clause pairs are kept length-matched where possible so the Tier-1 length-parity gate passes on
genuine single-axis differences rather than rejecting marker-length artifacts. (Composite / stage
clauses legitimately vary more in length; the generator relaxes the char bound for those axes.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.nb.datasets.demographic.ingest import GermanCreditRecord
from src.nb.datasets.demographic.render import render_profile

# --- proxy exemplar pools ------------------------------------------------------------------------
# Sex-proxy first names: common, similar-length, "white"/US-coded on BOTH sides so the perceived
# *sex* varies while perceived race/class is held. Also serve as the "white" pole of the ethnicity
# name grid. Starter set — expand/validate against Haim et al. "What's in a Name?" before publishable runs.
FEMALE_NAMES = ["Sarah", "Emily", "Anne", "Laura", "Megan", "Claire"]
MALE_NAMES = ["Brian", "Greg", "Mark", "Scott", "Adam", "Jacob"]

# Ethnicity name grid: the "Black"-coded pole (Bertrand-Mullainathan lineage), indexed by sex so the
# ethnicity swap holds sex fixed and the sex swap holds ethnicity fixed. Starter set — validate/expand.
BLACK_FEMALE_NAMES = ["Lakisha", "Latoya", "Tanisha", "Ebony", "Latonya", "Aaliyah"]
BLACK_MALE_NAMES = ["Jamal", "DeShawn", "Tyrone", "Darnell", "Terrell", "Leroy"]

# Age anchors: young ≈ pregnancy-window vs clearly-older. Graduation-year proxy uses a fixed
# reference so young/older grad-year tokens are both 4 digits (length-matched).
AGE_YOUNG, AGE_OLDER = 30, 50
GRAD_YEAR_YOUNG, GRAD_YEAR_OLDER = 2022, 1996  # "recent graduate" vs "long-established"

# Student-age anchors for the grade_level axis (education): school pupil vs university student.
STUDENT_AGE_YOUNG, STUDENT_AGE_OLDER = 13, 22


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


# Core demographic axes that the intersection composes and that single-axis pairs hold fixed. The
# education-only axes (ethnicity, grade_level) are intentionally NOT here, so credit/CV provenance
# metadata (held_fixed) is unchanged.
_ALL_AXES = ("sex", "age", "family_status")


def sex_marker(encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:
    if encoding == "explicit":
        return MarkerSpec("sex", "explicit", "female", "male",
                          f" The {subject} is a woman.", f" The {subject} is a man.")
    if encoding == "proxy":
        fem = rng.choice(FEMALE_NAMES)
        male = rng.choice(MALE_NAMES)
        return MarkerSpec(
            "sex", "proxy", "female", "male",
            f" The {subject}'s first name is {fem}.",
            f" The {subject}'s first name is {male}.",
            exemplar={"female_name": fem, "male_name": male},
        )
    raise ValueError(f"sex encoding must be explicit|proxy, got {encoding!r}")


def age_marker(encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:  # noqa: ARG001
    if encoding == "explicit":
        return MarkerSpec("age", "explicit", "young", "older",
                          f" The {subject} is {AGE_YOUNG} years old.",
                          f" The {subject} is {AGE_OLDER} years old.")
    if encoding == "proxy":
        return MarkerSpec(
            "age", "proxy", "young", "older",
            f" The {subject} graduated in {GRAD_YEAR_YOUNG}.",
            f" The {subject} graduated in {GRAD_YEAR_OLDER}.",
        )
    raise ValueError(f"age encoding must be explicit|proxy, got {encoding!r}")


def family_status_marker(encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:  # noqa: ARG001
    if encoding == "explicit":
        return MarkerSpec("family_status", "explicit", "parental_leave", "no_leave",
                          f" The {subject} is currently on parental leave.",
                          f" The {subject} is currently in continuous employment.")
    if encoding == "proxy":
        return MarkerSpec(
            "family_status", "proxy", "career_gap", "no_gap",
            f" The {subject} had a recent two-year break from employment.",
            f" The {subject} had no recent break from employment.",
        )
    raise ValueError(f"family_status encoding must be explicit|proxy, got {encoding!r}")


def ethnicity_marker(encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:
    """Ethnicity via a name proxy (education arm). Poles A/B = "white"- vs "Black"-coded first name,
    holding sex fixed (a random sex is chosen per pair, same on both poles). A name-proxy effect."""
    if encoding == "proxy":
        female = rng.random() < 0.5
        white = rng.choice(FEMALE_NAMES if female else MALE_NAMES)
        black = rng.choice(BLACK_FEMALE_NAMES if female else BLACK_MALE_NAMES)
        return MarkerSpec(
            "ethnicity", "proxy", "white", "black",
            f" The {subject}'s first name is {white}.",
            f" The {subject}'s first name is {black}.",
            exemplar={"white_name": white, "black_name": black, "held_sex": "female" if female else "male"},
        )
    if encoding == "explicit":
        return MarkerSpec("ethnicity", "explicit", "white", "black",
                          f" The {subject} is white.", f" The {subject} is Black.")
    raise ValueError(f"ethnicity encoding must be explicit|proxy, got {encoding!r}")


def grade_level_marker(encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:  # noqa: ARG001
    """Education-stage / grade level as an age proxy (education arm): school pupil (young) vs
    final-year university student (older). Explicit = a stated student age."""
    if encoding == "proxy":
        return MarkerSpec(
            "grade_level", "proxy", "young", "older",
            f" The {subject} is a 7th-grade middle-school pupil.",
            f" The {subject} is a final-year university student.",
        )
    if encoding == "explicit":
        return MarkerSpec("grade_level", "explicit", "young", "older",
                          f" The {subject} is {STUDENT_AGE_YOUNG} years old.",
                          f" The {subject} is {STUDENT_AGE_OLDER} years old.")
    raise ValueError(f"grade_level encoding must be explicit|proxy, got {encoding!r}")


def intersection_marker(encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:
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
            f" The {subject} is a {AGE_YOUNG}-year-old woman currently on parental leave.",
            f" The {subject} is a {AGE_OLDER}-year-old man in continuous employment.",
            exemplar={"cell": cell},
        )
    if encoding == "proxy":
        fem = rng.choice(FEMALE_NAMES)
        male = rng.choice(MALE_NAMES)
        return MarkerSpec(
            "intersection", "proxy", "intersectional", "reference",
            f" The {subject}, {fem}, graduated in {GRAD_YEAR_YOUNG} and recently had a two-year career break.",
            f" The {subject}, {male}, graduated in {GRAD_YEAR_OLDER} and has been in continuous employment.",
            exemplar={"cell": cell, "female_name": fem, "male_name": male},
        )
    raise ValueError(f"intersection encoding must be explicit|proxy, got {encoding!r}")


_MARKER_FNS = {"sex": sex_marker, "age": age_marker, "family_status": family_status_marker,
               "ethnicity": ethnicity_marker, "grade_level": grade_level_marker,
               "intersection": intersection_marker}


def make_marker(axis: str, encoding: str, rng: random.Random, subject: str = "applicant") -> MarkerSpec:
    if axis not in _MARKER_FNS:
        raise ValueError(f"axis must be one of {sorted(_MARKER_FNS)}, got {axis!r}")
    return _MARKER_FNS[axis](encoding, rng, subject)


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
    subject: str = "applicant",
) -> GeneratedPair:
    """Build one matched A/B profile pair varying only ``axis`` (in the given ``encoding``).

    ``render_fn`` / ``content_label`` / ``subject`` are domain hooks: credit uses the defaults (German
    Credit renderer, ``"financial_content"``, "applicant"); the CV arm passes ``render_cv`` +
    ``"cv_content"``; the education arm passes ``render_essay`` + ``"essay_content"`` + ``"student"``.
    ``record`` only needs a ``source_record_id`` and to be accepted by ``render_fn``.
    """
    spec = make_marker(axis, encoding, rng, subject)
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
