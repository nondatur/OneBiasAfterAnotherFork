#!/usr/bin/env python3
"""
Positioned-argument (A2) neutral-baseline decomposition.

The battery's auto-influence (reward A vs reward B) was large on *every* axis including the control, so it
cannot by itself separate "the RM prefers marginalized standpoints" from "the RM reacts to any appended
first-person persona". This script scores three variants per essay --- **neutral** (no positionality),
**pole_a** (marked identity), **pole_b** (reference identity) --- and reports the decomposition:

  delta_a   = mean reward(pole_a) - mean reward(neutral)   [does claiming identity A raise/lower reward?]
  delta_b   = mean reward(pole_b) - mean reward(neutral)
  main_fx   = mean(delta_a, delta_b)                       [effect of adding ANY standpoint sentence]
  identity_gap = delta_a - delta_b (= mean base_gap)       [the identity-specific part]
  auto_infl = 2*|P(reward_a > reward_b) - 0.5|

Read across axes: if the demographic `identity_gap` is large while the *genuinely neutral* controls
(pos_ctrl_hobby/pet/region) are ~0, the effect is demographic-specific. If the neutral controls also show a
large gap, it is generic persona-sensitivity. `main_fx` says whether the RM simply likes (or dislikes) a
personal-experience appeal regardless of who makes it.

Usage:
    python experiments/run_positioned_maineffect.py --config configs/demographic_edupos_qwen06.yaml \
        --source persuade --axes pos_sex,pos_race,pos_class,pos_origin,pos_intersection,\
pos_control,pos_ctrl_hobby,pos_ctrl_pet,pos_ctrl_region --n-essays 200 --position conclusion
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.datasets.base import ContrastivePair, format_conversation
from src.nb.datasets.demographic.domains import get_domain
from src.nb.datasets.demographic.edu_ingest import DEFAULT_ASAP_PATH, DEFAULT_PERSUADE_PATH, load_asap, load_persuade
from src.nb.datasets.demographic.positionality import IDENTITY_AXES, make_positioned_pair, render_neutral
from src.nb.experiments.base import ExperimentConfig
from src.nb.experiments.demographic import DemographicBiasExperiment
from src.nb.nullbias.probe import build_probe_direction, get_rewards_both

_LOADERS = {"persuade": (load_persuade, DEFAULT_PERSUADE_PATH), "asap": (load_asap, DEFAULT_ASAP_PATH)}


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def run_axis(exp, cfg, dom, essays, axis, position, rng) -> Dict[str, Any]:
    tok = exp.tokenizer
    fmt = lambda txt: format_conversation(tok, dom.assessment_prompt, txt)
    neutral, a, b, pairs = [], [], [], []
    for rec in essays:
        pair = make_positioned_pair(rec, axis, position, rng)
        neutral.append(fmt(render_neutral(rec)))
        a.append(fmt(pair.text_a))
        b.append(fmt(pair.text_b))
        pairs.append(ContrastivePair(positive_text=fmt(pair.text_a), negative_text=fmt(pair.text_b),
                                     metadata={"axis": axis}))
    probe, meta = build_probe_direction(exp.model, tok, pairs, batch_size=cfg.batch_size,
                                        device=cfg.device, max_length=cfg.max_length)
    n = len(essays)
    base, nulled = get_rewards_both(exp.model, tok, neutral + a + b, probe, batch_size=cfg.batch_size,
                                    device=cfg.device, max_length=cfg.max_length, null_alpha=1.0,
                                    show_progress=False)
    base = base.tolist()
    r_neu, r_a, r_b = base[:n], base[n:2 * n], base[2 * n:]
    delta_a = _mean([x - y for x, y in zip(r_a, r_neu)])
    delta_b = _mean([x - y for x, y in zip(r_b, r_neu)])
    pref_a = _mean([1.0 if x > y else 0.0 for x, y in zip(r_a, r_b)])
    la, lb, id_a, id_b = IDENTITY_AXES[axis]
    return {
        "axis": axis, "n": n, "probe_accuracy": meta.get("probe_accuracy"),
        "identity_a": id_a, "identity_b": id_b,
        "mean_neutral": _mean(r_neu), "mean_a": _mean(r_a), "mean_b": _mean(r_b),
        "delta_a": delta_a, "delta_b": delta_b,
        "main_effect": (delta_a + delta_b) / 2, "identity_gap": delta_a - delta_b,
        "auto_influence": 2 * abs(pref_a - 0.5), "pref_a": pref_a,
        "frac_a_above_neutral": _mean([1.0 if x > y else 0.0 for x, y in zip(r_a, r_neu)]),
        "frac_b_above_neutral": _mean([1.0 if x > y else 0.0 for x, y in zip(r_b, r_neu)]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/demographic_edupos_qwen06.yaml"))
    ap.add_argument("--source", choices=sorted(_LOADERS), default="persuade")
    ap.add_argument("--raw-path", default=None)
    ap.add_argument("--axes", default=",".join(IDENTITY_AXES))
    ap.add_argument("--position", default="conclusion")
    ap.add_argument("--n-essays", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/results/demographic/maineffect_edupos_persuade_qwen06.json"))
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    dom = get_domain(cfg.extra.get("domain", "education"))
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    loader, default_path = _LOADERS[args.source]
    essays = loader(args.raw_path or default_path, n=args.n_essays, seed=args.seed)

    exp = DemographicBiasExperiment(cfg)
    exp.load_model()
    rng = random.Random(args.seed)
    results = [run_axis(exp, cfg, dom, essays, ax, args.position, rng) for ax in axes]
    for ax in axes:
        print(f"[maineffect] {ax} done", flush=True)

    print("\n" + "=" * 100)
    print(f"POSITIONED MAIN-EFFECT [{args.source}/{args.position}] — {cfg.model_path} (n={len(essays)})")
    print("=" * 100)
    print(f"{'axis':18} {'mean_neu':>8} {'delta_a':>8} {'delta_b':>8} {'main_fx':>8} {'id_gap':>8} {'auto_AI':>8}")
    for r in results:
        print(f"{r['axis']:18} {r['mean_neutral']:>8.3f} {r['delta_a']:>8.3f} {r['delta_b']:>8.3f} "
              f"{r['main_effect']:>8.3f} {r['identity_gap']:>8.3f} {r['auto_influence']:>8.3f}")
    print("=" * 100)
    print("id_gap on demographic axes vs (near-0 on) pos_ctrl_* ⇒ demographic-specific; else persona-sensitivity.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": cfg.model_path, "source": args.source,
                                    "position": args.position, "results": results}, indent=2))
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
