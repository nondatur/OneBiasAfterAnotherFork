#!/usr/bin/env python3
"""
Real-field family-status arm (external-validity cross-check), credit arm, one RM (default Qwen3-0.6B).

Uses German Credit's ACTUAL `personal_status_sex` field (neutralized in the synthetic arm) to read a
real-data marital/family-status signal. Contrast holds sex = male (females are all one bucket):
    single males (A93)  vs  ever-partnered males (married/widowed A94 + divorced/separated A91).

Builds the real-field marital difference-of-means direction and reports:
  1. cross-check cosines vs the SYNTHETIC family-status and sex directions (does the real field encode
     the same direction as the synthetic injection? does the known sex/marital entanglement surface?);
  2. the single-vs-partnered mean reward gap, baseline vs nulled (project out the real-field direction).

HONEST LIMITATION: the two marital groups are different applicants whose financials also differ, so
this direction/gap is **confounded** with marital-correlated financials — a cross-check, not a clean
single-axis result. (Deferred refinement: balance/match the groups on financials.)

Usage:
    python experiments/run_realfield_familystatus.py --config configs/demographic_credit_sex_qwen06.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.nb.datasets.base import ContrastivePair, format_conversation
from src.nb.datasets.demographic.dataset import ASSESSMENT_PROMPT, CreditDemographicDataset
from src.nb.datasets.demographic.ingest import load_german_credit
from src.nb.datasets.demographic.markers import real_field_clause
from src.nb.datasets.demographic.render import render_profile, TEMPLATES
from src.nb.experiments.base import ExperimentConfig
from src.nb.experiments.demographic import DemographicBiasExperiment
from src.nb.nullbias.probe import build_probe_direction, get_rewards_both


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a / (a.norm() + 1e-8)) @ (b / (b.norm() + 1e-8)))


def _synthetic_dir(exp, cfg, axis):
    ds = CreditDemographicDataset(cfg.dataset_source, axis=axis, encoding="explicit",
                                  probe_size=cfg.probe_size, split_seed=cfg.split_seed)
    probe, _ = build_probe_direction(exp.model, exp.tokenizer, ds.get_probe_pairs(exp.tokenizer),
                                     batch_size=cfg.batch_size, device=cfg.device, max_length=cfg.max_length)
    return probe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_credit_sex_qwen06.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/results/demographic/realfield_familystatus_qwen06.json"))
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    tok = exp.tokenizer
    fmt = lambda r, t: format_conversation(tok, ASSESSMENT_PROMPT,
                                           render_profile(r, t, marker=real_field_clause(r)))

    # Select males, split by real marital status; balance groups (sample singles to partnered count).
    males = [r for r in load_german_credit() if r.raw_sex == "male"]
    single = [r for r in males if r.raw_marital == "single"]
    partnered = [r for r in males if r.raw_marital in ("married/widowed", "divorced/separated")]
    rng = random.Random(args.seed)
    rng.shuffle(single)
    n = min(len(single), len(partnered))
    single, partnered = single[:n], partnered[:n]
    tids = sorted(TEMPLATES)
    single_txt = [fmt(r, tids[i % len(tids)]) for i, r in enumerate(single)]
    partnered_txt = [fmt(r, tids[i % len(tids)]) for i, r in enumerate(partnered)]

    # Real-field marital direction (single − partnered); DiffMean uses only group means.
    pairs = [ContrastivePair(positive_text=s, negative_text=p) for s, p in zip(single_txt, partnered_txt)]
    real_dir, meta = build_probe_direction(exp.model, tok, pairs, batch_size=cfg.batch_size,
                                           device=cfg.device, max_length=cfg.max_length)

    # Cross-check cosines vs synthetic directions.
    fam_dir = _synthetic_dir(exp, cfg, "family_status")
    sex_dir = _synthetic_dir(exp, cfg, "sex")
    cos_fam, cos_sex = _cos(real_dir, fam_dir), _cos(real_dir, sex_dir)

    # Group reward gap (single − partnered), baseline vs nulled (project out real_dir).
    s_base, s_null = get_rewards_both(exp.model, tok, single_txt, real_dir, batch_size=cfg.batch_size,
                                      device=cfg.device, max_length=cfg.max_length, null_alpha=1.0,
                                      show_progress=False)
    p_base, p_null = get_rewards_both(exp.model, tok, partnered_txt, real_dir, batch_size=cfg.batch_size,
                                      device=cfg.device, max_length=cfg.max_length, null_alpha=1.0,
                                      show_progress=False)
    gap_base = float(s_base.mean() - p_base.mean())
    gap_null = float(s_null.mean() - p_null.mean())

    print("\n" + "=" * 78)
    print(f"REAL-FIELD FAMILY-STATUS (marital) — {cfg.model_path}")
    print("=" * 78)
    print(f"groups: single males n={n}  vs  ever-partnered males n={n}  (sex held = male)")
    print(f"real-field probe: accuracy={meta.get('probe_accuracy', 0):.2%}  separation={meta.get('separation', 0):.3f}")
    print(f"cosine(real_marital, synthetic FAMILY-STATUS dir) = {cos_fam:+.4f}   (external validity)")
    print(f"cosine(real_marital, synthetic SEX dir)           = {cos_sex:+.4f}   (sex/marital entanglement)")
    print(f"reward gap single−partnered:  baseline={gap_base:+.4f}   nulled={gap_null:+.4f}")
    print("=" * 78)
    print("Caveat: groups differ in financials too → confounded cross-check, not clean single-axis.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": cfg.model_path, "n_per_group": n, "seed": args.seed,
        "probe_accuracy": meta.get("probe_accuracy"), "probe_separation": meta.get("separation"),
        "cosine_real_vs_synthetic_family": cos_fam, "cosine_real_vs_synthetic_sex": cos_sex,
        "reward_gap_single_minus_partnered": {"baseline": gap_base, "nulled": gap_null},
        "single_ids": [r.source_record_id for r in single],
        "partnered_ids": [r.source_record_id for r in partnered],
    }, indent=2))
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
