"""
Domain registry for the demographic-bias arms.

A single source of truth for everything that differs between the **credit**, **hiring (CV-screening)**
and **education (grading)** domains: the matched-pair dataset class + default manifest, the neutral
renderer + framing prompt + template ids, how to load the underlying records, and which records count
as the "stronger" applicant (the quality ground truth for cross-influence). All three now sit on real
substrates. Runners (`additivity_check`, `run_demographic_battery`,
`run_crossinfluence`) and `DemographicBiasExperiment._create_dataset` all resolve a `DomainSpec` from
`cfg.extra["domain"]` (or a `--domain` flag) so adding a domain is one entry here.

Imports only from the datasets package (no `experiments` import) → no import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from src.nb.datasets.demographic.dataset import ASSESSMENT_PROMPT, CreditDemographicDataset
from src.nb.datasets.demographic.bios_dataset import BIOS_ASSESSMENT_PROMPT, BiosDemographicDataset
from src.nb.datasets.demographic.edu_dataset import EDU_ASSESSMENT_PROMPT, EducationDemographicDataset
from src.nb.datasets.demographic.ingest import load_german_credit
from src.nb.datasets.demographic.bios_ingest import DEFAULT_BIOS_PATH, load_bias_in_bios
from src.nb.datasets.demographic.edu_ingest import DEFAULT_PERSUADE_PATH, load_persuade
from src.nb.datasets.demographic.render import TEMPLATES, render_profile
from src.nb.datasets.demographic.bios_render import BIOS_TEMPLATES, render_bio
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

# Hiring arm. The substrate is REAL biographies (Bias-in-Bios), not the synthetic CV generator that
# used to back this domain — `cv_ingest.generate_candidates` is retained for reproducing pre-2026-08
# results but is no longer what `domain=cv` loads. `qualified` here means role-match
# (profession == target_role), so it is a genuine quality axis rather than an invented heuristic.
CV = DomainSpec(
    name="cv",
    dataset_cls=BiosDemographicDataset,
    default_pairs="data/demographic/cv/pairs.jsonl",
    render_fn=render_bio,
    assessment_prompt=BIOS_ASSESSMENT_PROMPT,
    template_ids=tuple(sorted(BIOS_TEMPLATES)),
    # Real biographies are user-downloaded; raises with fetch instructions if absent.
    load_records=lambda: load_bias_in_bios(DEFAULT_BIOS_PATH),
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
