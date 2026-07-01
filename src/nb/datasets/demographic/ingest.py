"""
Ingest UCI German Credit (Statlog, CC-BY-4.0) and decode the A-coded fields into a typed record.

Source file: `data/demographic/credit/raw/german.data` — 1000 space-separated rows, 20 attributes
+ class. The A-code → human-readable mapping below is the documented codebook (german.doc).

We deliberately separate the **financial slots** (used to render a demographically-neutral profile)
from the two protected/source fields — `personal_status_sex` (attr 9) and `age_years` (attr 13) —
which are preserved on the record (prefixed `raw_`) for the optional real-field arm but are NOT
rendered into the neutral baseline (demographics are injected synthetically downstream).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# --- A-code → readable value maps (from german.doc) ---------------------------------------------
CHECKING = {"A11": "less than 0 EUR", "A12": "0 to 199 EUR", "A13": "200 EUR or more",
            "A14": "no checking account"}
CREDIT_HISTORY = {
    "A30": "no credits taken / all paid back duly", "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly until now", "A33": "delay in paying off in the past",
    "A34": "critical account / other credits existing elsewhere",
}
PURPOSE = {"A40": "a new car", "A41": "a used car", "A42": "furniture or equipment",
           "A43": "radio or television", "A44": "domestic appliances", "A45": "repairs",
           "A46": "education", "A48": "retraining", "A49": "business", "A410": "other purposes"}
SAVINGS = {"A61": "less than 100 EUR", "A62": "100 to 499 EUR", "A63": "500 to 999 EUR",
           "A64": "1000 EUR or more", "A65": "unknown / no savings account"}
EMPLOYMENT = {"A71": "unemployed", "A72": "less than 1 year", "A73": "1 to 3 years",
              "A74": "4 to 6 years", "A75": "7 years or more"}
PROPERTY = {"A121": "real estate", "A122": "building-society savings or life insurance",
            "A123": "a car or other property", "A124": "no known property"}
OTHER_PLANS = {"A141": "at another bank", "A142": "at stores", "A143": "none"}
HOUSING = {"A151": "rented", "A152": "owned", "A153": "provided for free"}
JOB = {"A171": "unemployed or unskilled (non-resident)", "A172": "unskilled resident",
       "A173": "a skilled employee or official",
       "A174": "in management / self-employed / highly qualified"}
# protected/source fields (preserved, not rendered into the neutral baseline)
PERSONAL_STATUS_SEX = {
    "A91": ("male", "divorced/separated"), "A92": ("female", "divorced/separated/married"),
    "A93": ("male", "single"), "A94": ("male", "married/widowed"), "A95": ("female", "single"),
}

_COLUMNS = [
    "checking", "duration_months", "credit_history", "purpose", "credit_amount_dm", "savings",
    "employment_since", "installment_rate_pct", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age_years", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "credit_class",
]


@dataclass
class GermanCreditRecord:
    """One decoded German Credit applicant. Financial slots feed the neutral renderer; the two
    protected fields (`raw_sex`, `raw_marital`, `raw_age_years`) are preserved for the real-field
    arm / quality signal but are not part of the demographically-neutral profile."""

    source_record_id: str
    # financial slots (rendered)
    checking: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount_dm: int
    savings: str
    employment_since: str
    installment_rate_pct: int
    property: str
    other_installment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    telephone: bool
    foreign_worker: bool
    credit_good: bool  # class 1 = good credit (the ground-truth quality label)
    # protected / source fields (preserved, NOT rendered in the neutral baseline)
    raw_sex: str
    raw_marital: str
    raw_age_years: int
    extra: Dict[str, object] = field(default_factory=dict)


DEFAULT_RAW_PATH = Path("data/demographic/credit/raw/german.data")


def load_german_credit(path: Path | str = DEFAULT_RAW_PATH) -> List[GermanCreditRecord]:
    """Parse `german.data` into decoded :class:`GermanCreditRecord` objects (order preserved)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"German Credit raw file not found at {path}. Download it first, e.g.:\n"
            "  curl -L -o data/demographic/credit/raw/german.data \\\n"
            "    https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        )

    records: List[GermanCreditRecord] = []
    for i, line in enumerate(path.read_text().splitlines()):
        tok = line.split()
        if len(tok) != len(_COLUMNS):
            continue
        row = dict(zip(_COLUMNS, tok))
        sex, marital = PERSONAL_STATUS_SEX[row["personal_status_sex"]]
        records.append(
            GermanCreditRecord(
                source_record_id=f"german-{i:04d}",
                checking=CHECKING[row["checking"]],
                duration_months=int(row["duration_months"]),
                credit_history=CREDIT_HISTORY[row["credit_history"]],
                purpose=PURPOSE[row["purpose"]],
                credit_amount_dm=int(row["credit_amount_dm"]),
                savings=SAVINGS[row["savings"]],
                employment_since=EMPLOYMENT[row["employment_since"]],
                installment_rate_pct=int(row["installment_rate_pct"]),
                property=PROPERTY[row["property"]],
                other_installment_plans=OTHER_PLANS[row["other_installment_plans"]],
                housing=HOUSING[row["housing"]],
                existing_credits=int(row["existing_credits"]),
                job=JOB[row["job"]],
                num_dependents=int(row["num_dependents"]),
                telephone=(row["telephone"] == "A192"),
                foreign_worker=(row["foreign_worker"] == "A201"),
                credit_good=(row["credit_class"] == "1"),
                raw_sex=sex,
                raw_marital=marital,
                raw_age_years=int(row["age_years"]),
            )
        )
    if not records:
        raise ValueError(f"No valid German Credit rows parsed from {path}.")
    return records
