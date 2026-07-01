"""
Demographic / intersectional protected-attribute bias — data-creation pipeline (Direction 1).

Builds matched, single-axis **contrastive pairs** (for the difference-of-means probe) and matched
**applicant-pair eval items** (for cross-/auto-influence) on a credit-scoring substrate (UCI German
Credit). Demographics are injected as controlled templated markers so each comparison varies only
one axis and holds financial content fixed (the "hold-everything-fixed" design grammar).

Stages: ingest+render (German Credit → neutral NL credit profile) → marker injection (sex/age/
family-status, explicit + proxy) → Tier-1 structural validation gate → manifest writer.

These outputs plug into the existing probe machinery (`src.nb.nullbias.probe.build_probe_direction`
+ `project_to_null_space`) via a `ProbeDataset` loader — no changes to that machinery.
"""

from src.nb.datasets.demographic.ingest import GermanCreditRecord, load_german_credit
from src.nb.datasets.demographic.render import render_profile, TEMPLATES
from src.nb.datasets.demographic.dataset import CreditDemographicDataset

__all__ = [
    "GermanCreditRecord",
    "load_german_credit",
    "render_profile",
    "TEMPLATES",
    "CreditDemographicDataset",
]
