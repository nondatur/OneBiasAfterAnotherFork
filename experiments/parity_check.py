#!/usr/bin/env python3
"""
MLX vs. transformers numerical-parity check.

Runs the SAME tiny experiment (probe build + evaluation) twice on the same
inputs — once with a **CPU float32 transformers** reference model and once with
the **MLX (bf16)** backend — and compares:

  * reward scores      → Pearson r        (gate: >= --r-min, default 0.99)
  * probe direction(s) → subspace overlap (gate: >= --cos-min, default 0.99)
  * bias metrics       → max |Δ|          (gate: <= --metric-tol, default 0.02)

MLX is a development/prototyping path; publishable numbers should still be
validated on full-precision CUDA. Exits non-zero if any gate fails.

Usage:
    python experiments/parity_check.py \
        --config configs/position_skywork_qwen06_gsm8k.yaml \
        --max-examples 50 --probe-size 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.nb.experiments.base import BiasExperiment, ExperimentConfig
from src.nb.nullbias.probe import gram_schmidt, get_rewards_both

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("parity")


def _select_experiment_class(config: ExperimentConfig):
    """Mirror experiments/run_experiment.py's class selection."""
    from src.nb.experiments.length import LengthBiasExperiment
    from src.nb.experiments.sycophancy import SycophancyBiasExperiment
    from src.nb.experiments.uncertainty import UncertaintyBiasExperiment
    from src.nb.experiments.position import (
        PositionBiasExperiment,
        BinaryPositionBiasExperiment,
        FreeformPositionBiasExperiment,
    )

    classes = {
        "length": LengthBiasExperiment,
        "sycophancy": SycophancyBiasExperiment,
        "uncertainty": UncertaintyBiasExperiment,
        "position": PositionBiasExperiment,
    }
    cls = classes.get(config.bias_type)
    if cls is None:
        raise ValueError(f"Unknown bias type: {config.bias_type}")
    if config.bias_type == "position":
        dc = config.dataset_class
        if dc in ("position_freeform", "position_freeform_bigbench", "position_freeform_plausibleqa"):
            cls = FreeformPositionBiasExperiment
        elif dc == "position_plausibleqa" or ("plausibleqa" in config.dataset_source.lower() and not dc):
            cls = BinaryPositionBiasExperiment
    return cls


def _load_reference_model(model_path: str, trust_remote_code: bool, ref_dtype: str = "float32"):
    """Load the HF reward model on CPU (the parity reference)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[ref_dtype]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # identical tokenization to the MLX backend
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, dtype=dtype, device_map="cpu"
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def _run_pipeline(config: ExperimentConfig, model: Any, tokenizer: Any):
    """Build probe + evaluate using the shared experiment machinery.

    Returns (probe, baseline_rewards_flat, baseline_metrics, nulled_metrics, n_eval).
    """
    exp: BiasExperiment = _select_experiment_class(config)(config)
    exp.model = model
    exp.tokenizer = tokenizer
    exp.load_dataset()
    exp.build_probe()                       # populates exp.probe
    eval_examples = exp.dataset.get_eval_examples(exp.tokenizer)
    all_texts, text_meta = exp._get_all_texts_and_variants(eval_examples)
    baseline, nulled = get_rewards_both(
        model=exp.model, tokenizer=exp.tokenizer, texts=all_texts, probe=exp.probe,
        batch_size=config.batch_size, device=config.device, max_length=config.max_length,
        null_alpha=config.null_alpha, show_progress=False,
    )
    n = len(eval_examples)
    base_metrics = exp._compute_metrics(exp._organize_rewards(baseline, text_meta, n), eval_examples)
    nulled_metrics = exp._compute_metrics(exp._organize_rewards(nulled, text_meta, n), eval_examples)
    return exp.probe, baseline, base_metrics, nulled_metrics, n


def _normalize_metric(key: str, value: float) -> float:
    """Map a metric to a comparable [0, 1]-ish scale for tolerance checks.

    - percentage-valued metrics (``*_pct``, ``max_position_bias``) → /100
    - everything else returned as-is (accuracies/rates are already in [0, 1])
    """
    if key.endswith("_pct") or key == "max_position_bias":
        return value / 100.0
    return value


def _is_count_metric(key: str) -> bool:
    """Counts are fixed by the (shared) dataset split, not by model fidelity."""
    return key.startswith("n_") or key == "num_choices"


def _metric_deltas(ref: Dict[str, float], cand: Dict[str, float]) -> Dict[str, float]:
    """Normalized |Δ| per shared numeric, non-count metric."""
    out: Dict[str, float] = {}
    for k in sorted(ref):
        if _is_count_metric(k):
            continue
        rv, cv = ref.get(k), cand.get(k)
        if isinstance(rv, (int, float)) and isinstance(cv, (int, float)):
            out[k] = abs(_normalize_metric(k, float(rv)) - _normalize_metric(k, float(cv)))
    return out


def _is_diagnostic_metric(key: str) -> bool:
    """Per-bucket sub-statistics: reported, but not part of the PASS/FAIL gate.

    These (e.g. ``accuracy_when_A`` over a handful of examples, ``position_C_pct``)
    quantize at 1/N_bucket, so on small samples they swing by a single argmax flip.
    The headline aggregates (overall ``accuracy``, ``max_position_bias``,
    ``*_gap``, ``preference_accuracy``, …) are gated instead.
    """
    k = key.lower()
    if "_when_" in k or "_at_" in k:
        return True
    if k.startswith("position_") and k.endswith("_pct"):
        return True
    return False


def _significant_overlap(svals, floor: float = 0.5) -> float:
    """Mean principal-angle cosine over *significant* probe directions.

    Multi-vector probes (e.g. position A/B/C/D-vs-rest) are effectively
    rank-deficient: after correctness-cleaning + Gram-Schmidt, the last
    orthonormal direction carries ~zero signal, so its orientation is
    precision noise in BOTH runs and contributes a near-zero singular value.
    Averaging it in (the plain mean) is misleading; we average only directions
    whose overlap clears ``floor``. For a single-vector probe this is just the
    (only) cosine.
    """
    sig = [s for s in svals if s >= floor]
    return (sum(sig) / len(sig)) if sig else 0.0


def _subspace_overlap(a: torch.Tensor, b: torch.Tensor):
    """Mean cosine of principal angles between span(a) and span(b), in [0, 1].

    For 1-D probes this is |cos(angle)|; for [k, d] bases it measures how aligned
    the two subspaces are (1.0 == identical subspace).
    """
    a = a.float().reshape(1, -1) if a.dim() == 1 else a.float()
    b = b.float().reshape(1, -1) if b.dim() == 1 else b.float()
    qa = gram_schmidt([v for v in a])
    qb = gram_schmidt([v for v in b])
    if qa.shape[0] == 0 or qb.shape[0] == 0:
        return 0.0, []
    svals = torch.linalg.svdvals(qa @ qb.T)
    return float(svals.mean().clamp(0.0, 1.0)), [round(float(s), 4) for s in svals]


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    xc = x - x.mean()
    yc = y - y.mean()
    denom = xc.norm() * yc.norm()
    if denom < 1e-12:
        return float("nan")
    return float((xc @ yc) / denom)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="Experiment config YAML")
    ap.add_argument("--max-examples", type=int, default=50, help="Cap eval examples for speed")
    ap.add_argument("--probe-size", type=int, default=100, help="Probe training size")
    ap.add_argument("--r-min", type=float, default=0.99, help="Min reward Pearson r")
    ap.add_argument("--cos-min", type=float, default=0.99, help="Min probe subspace overlap")
    ap.add_argument("--metric-tol", type=float, default=0.02, help="Max |Δ| per headline bias metric")
    ap.add_argument("--mlx-quant", type=str, default=None, choices=["4bit", "8bit"])
    ap.add_argument("--ref-dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"],
                    help="Reference (transformers/CPU) dtype. Use bfloat16 for large models that "
                         "don't fit in RAM at fp32 (e.g. 8B on a 32 GB Mac).")
    args = ap.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    config.max_test_examples = args.max_examples
    config.probe_size = args.probe_size
    config.save_probe = False

    # ---- Reference: CPU transformers (float32 by default) ----
    logger.info("[reference] loading %s on CPU %s ...", config.model_path, args.ref_dtype)
    config.device = "cpu"
    ref_model, ref_tok = _load_reference_model(config.model_path, config.trust_remote_code, args.ref_dtype)
    probe_ref, rewards_ref, base_ref, null_ref, n_ref = _run_pipeline(config, ref_model, ref_tok)
    del ref_model

    # ---- Candidate: MLX bf16 ----
    logger.info("[mlx] loading %s via MLX (%s) ...", config.model_path, args.mlx_quant or "bf16")
    from src.nb.backends.mlx_backend import MLXBackend

    config.device = "mlx"
    config.mlx_quant = args.mlx_quant
    mlx_backend = MLXBackend(
        model_path=config.model_path, trust_remote_code=config.trust_remote_code,
        mlx_quant=args.mlx_quant, max_length=config.max_length,
    )
    probe_mlx, rewards_mlx, base_mlx, null_mlx, n_mlx = _run_pipeline(config, mlx_backend, mlx_backend.tokenizer)

    # ---- Compare ----
    assert n_ref == n_mlx, f"eval count mismatch: {n_ref} vs {n_mlx}"
    r = _pearson(rewards_ref, rewards_mlx)
    max_abs = float((rewards_ref.float() - rewards_mlx.float()).abs().max())
    cos_mean, svals = _subspace_overlap(probe_ref, probe_mlx)
    cos_sig = _significant_overlap(svals)

    base_deltas = _metric_deltas(base_ref, base_mlx)
    null_deltas = _metric_deltas(null_ref, null_mlx)
    # Gate only on headline aggregates; per-bucket sub-stats are reported, not gated.
    gated_base = {k: v for k, v in base_deltas.items() if not _is_diagnostic_metric(k)}
    gated_null = {k: v for k, v in null_deltas.items() if not _is_diagnostic_metric(k)}
    max_base = max(gated_base.values()) if gated_base else 0.0
    max_null = max(gated_null.values()) if gated_null else 0.0

    def _section(title, ref, cand, deltas):
        print(f"\n  [{title}]  (counts excluded, %→fraction; * = diagnostic, not gated)")
        for k in sorted(deltas):
            diag = _is_diagnostic_metric(k)
            tag = " *" if diag else ""
            flag = "  <<" if (not diag and deltas[k] > args.metric_tol) else ""
            print(f"    {k:30s}{tag:2s} ref={float(ref[k]):+.4f}  mlx={float(cand[k]):+.4f}  |Δ|={deltas[k]:.4f}{flag}")

    print("\n" + "=" * 70)
    print(f"PARITY REPORT — {config.name}  (n_eval={n_ref}, probe_size={config.probe_size}, "
          f"mlx={args.mlx_quant or 'bf16'}, ref={args.ref_dtype})")
    print("=" * 70)
    print(f"reward Pearson r          : {r:.5f}   (gate >= {args.r_min})")
    print(f"reward max |Δ|            : {max_abs:.5f}")
    print(f"probe overlap (significant): {cos_sig:.5f}   (gate >= {args.cos_min})")
    print(f"probe overlap (mean)      : {cos_mean:.5f}   svals={svals}")
    print(f"max baseline metric |Δ|   : {max_base:.5f}   (gate <= {args.metric_tol}, headline only)")
    print(f"max nulled metric |Δ|     : {max_null:.5f}   (gate <= {args.metric_tol}, headline only)")
    _section("baseline", base_ref, base_mlx, base_deltas)
    _section("nulled", null_ref, null_mlx, null_deltas)
    print("=" * 70)

    reward_ok = r >= args.r_min
    metric_ok = (max_base <= args.metric_tol) and (max_null <= args.metric_tol)
    probe_ok = cos_sig >= args.cos_min
    print(f"reward fidelity : {'PASS' if reward_ok else 'FAIL'}")
    print(f"bias metrics    : {'PASS' if metric_ok else 'FAIL'}  (headline aggregates)")
    print(f"probe alignment : {'PASS' if probe_ok else 'FAIL'}  (significant subspace)")
    ok = reward_ok and metric_ok and probe_ok
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
