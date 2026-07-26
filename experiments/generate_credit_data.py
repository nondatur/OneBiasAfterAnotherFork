#!/usr/bin/env python3
"""
Generate the demographic credit-bias matched-pair dataset (credit arm).

Pipeline: load German Credit → render neutral profiles → inject single-axis demographic markers
(sex/age/family-status; explicit + proxy) → Tier-1 structural validation gate → write manifest.

Usage:
    python experiments/generate_credit_data.py \
        --axes sex,age,family_status --encodings explicit,proxy \
        --n-per 500 --seed 42 --out-dir data/demographic/credit
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.datasets.demographic.ingest import load_german_credit
from src.nb.datasets.demographic.markers import make_pair
from src.nb.datasets.demographic.render import TEMPLATES
from src.nb.datasets.demographic.validate import Thresholds, validate_pair
from src.nb.datasets.demographic.manifest import pair_to_record, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("gen-credit")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="sex,age,family_status")
    ap.add_argument("--encodings", default="explicit,proxy")
    ap.add_argument("--templates", default=",".join(sorted(TEMPLATES)))
    ap.add_argument("--n-per", type=int, default=500, help="Target clean pairs per (axis, encoding)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data/demographic/credit"))
    ap.add_argument("--raw", type=Path, default=Path("data/demographic/credit/raw/german.data"))
    ap.add_argument("--max-char-delta", type=int, default=12)
    ap.add_argument("--max-token-delta", type=int, default=3)
    ap.add_argument("--max-flesch-delta", type=float, default=8.0)
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    thr = Thresholds(args.max_char_delta, args.max_token_delta, args.max_flesch_delta)

    records = load_german_credit(args.raw)
    logger.info("Loaded %d German Credit records; templates=%s", len(records), templates)

    out_records: List[Dict[str, Any]] = []
    discards: Dict[str, Any] = {}

    for axis in axes:
        for enc in encodings:
            rng = random.Random(hash((args.seed, axis, enc)) & 0xFFFFFFFF)
            # Candidate order: shuffled (record × template) combos for variety / over-generation.
            combos = [(r, t) for r in records for t in templates]
            rng.shuffle(combos)
            kept, n_fail = 0, 0
            fail_reasons: Dict[str, int] = {}
            for rec, tid in combos:
                if kept >= args.n_per:
                    break
                pair = make_pair(rec, tid, axis, enc, rng)
                res = validate_pair(pair, thr)
                if not res.ok:
                    n_fail += 1
                    for rsn in res.reasons:
                        k = rsn.split(" (")[0].split(" >")[0]
                        fail_reasons[k] = fail_reasons.get(k, 0) + 1
                    continue
                item_id = f"credit-{axis}-{enc}-{tid}-{rec.source_record_id}"
                out_records.append(pair_to_record(pair, item_id, role="probe", seed=args.seed))
                kept += 1
            seen = kept + n_fail
            discards[f"{axis}/{enc}"] = {
                "kept": kept, "examined": seen,
                "discard_rate": round(n_fail / max(seen, 1), 4),
                "failure_reasons": fail_reasons,
            }
            logger.info("%s/%s: kept %d (examined %d, discard %.2f)",
                        axis, enc, kept, seen, n_fail / max(seen, 1))
            if kept < args.n_per:
                logger.warning("%s/%s: only %d/%d clean pairs available", axis, enc, kept, args.n_per)

    paths = write_manifest(
        args.out_dir, out_records, seed=args.seed, discard_report=discards,
        thresholds={"max_char_delta": thr.max_char_delta, "max_token_delta": thr.max_token_delta,
                    "max_flesch_delta": thr.max_flesch_delta},
    )
    logger.info("Wrote %d records → %s", len(out_records), paths["pairs"])
    logger.info("Manifest: %s | spot-check: %s", paths["manifest"], paths["spotcheck"])


if __name__ == "__main__":
    main()
