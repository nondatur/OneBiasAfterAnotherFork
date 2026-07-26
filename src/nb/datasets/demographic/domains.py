"""
Domain registry for the demographic-bias arms.

A single source of truth for everything that differs between the **credit** and **CV-screening**
domains: the matched-pair dataset class + default manifest, the neutral renderer + framing prompt +
template ids, how to load the underlying records, and which records count as the "stronger" applicant
(the quality ground truth for cross-influence). Runners (`additivity_check`, `run_demographic_battery`,
`run_crossinfluence`) and `DemographicBiasExperiment._create_dataset` all resolve a `DomainSpec` from
`cfg.extra["domain"]` (or a `--domain` flag) so adding a domain is one entry here.

Imports only from the datasets package (no `experiments` import) → no import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from src.nb.datasets.demographic.dataset import ASSESSMENT_PROMPT, CreditDemographicDataset
from src.nb.datasets.demographic.cv_dataset import CV_ASSESSMENT_PROMPT, CVDemographicDataset
from src.nb.datasets.demographic.edu_dataset import EDU_ASSESSMENT_PROMPT, EducationDemographicDataset
from src.nb.datasets.demographic.ingest import load_german_credit
from src.nb.datasets.demographic.cv_ingest import generate_candidates
from src.nb.datasets.demographic.edu_ingest import DEFAULT_PERSUADE_PATH, load_persuade
from src.nb.datasets.demographic.render import TEMPLATES, render_profile
from src.nb.datasets.demographic.cv_render import CV_TEMPLATES, render_cv
from src.nb.datasets.demographic.edu_render import EDU_TEMPLATES, render_essay


@dataclass(frozen=True)
class DomainSpec:
    """Everything a runner needs to operate one domain's direct-scoring arm."""

    name: str
    dataset_cls: type
    default_pairs: str
    render_fn: Callable[..., str]          # (record, template_id, marker="") -> str
    assessment_prompt: str
    template_ids: Tuple[str, ...]
    load_records: Callable[[], List[Any]]  # () -> records (each with .source_record_id)
    is_strong: Callable[[Any], bool]       # record -> True if the "stronger" applicant


CREDIT = DomainSpec(
    name="credit",
    dataset_cls=CreditDemographicDataset,
    default_pairs="data/demographic/credit/pairs.jsonl",
    render_fn=render_profile,
    assessment_prompt=ASSESSMENT_PROMPT,
    template_ids=tuple(sorted(TEMPLATES)),
    load_records=load_german_credit,
    is_strong=lambda r: r.credit_good,
)

CV = DomainSpec(
    name="cv",
    dataset_cls=CVDemographicDataset,
    default_pairs="data/demographic/cv/pairs.jsonl",
    render_fn=render_cv,
    assessment_prompt=CV_ASSESSMENT_PROMPT,
    template_ids=tuple(sorted(CV_TEMPLATES)),
    load_records=lambda: generate_candidates(1000, seed=42),
    is_strong=lambda r: r.qualified,
)

EDUCATION = DomainSpec(
    name="education",
    dataset_cls=EducationDemographicDataset,
    default_pairs="data/demographic/education/persuade/pairs.jsonl",
    render_fn=render_essay,
    assessment_prompt=EDU_ASSESSMENT_PROMPT,
    template_ids=tuple(sorted(EDU_TEMPLATES)),
    # Real essays are user-downloaded; load the PERSUADE corpus (raises with instructions if absent).
    load_records=lambda: load_persuade(DEFAULT_PERSUADE_PATH),
    is_strong=lambda r: r.high_quality,
)

DOMAINS = {CREDIT.name: CREDIT, CV.name: CV, EDUCATION.name: EDUCATION}


def get_domain(name: str) -> DomainSpec:
    if name not in DOMAINS:
        raise ValueError(f"domain must be one of {sorted(DOMAINS)}, got {name!r}")
    return DOMAINS[name]
