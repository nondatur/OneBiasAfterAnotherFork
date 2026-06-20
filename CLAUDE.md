# CLAUDE.md

Persistent context for Claude Code working in this repository.

> The **Repository layout** and **Architecture** sections below are seeded from
> the README. Verify and expand them by reading the actual source before relying
> on them — module names and function signatures have not been confirmed against
> the code.

## Project

**Mechanistic Reward Shaping** — research codebase (fork of
`drfein/OneBiasAfterAnother`, accompanying arXiv:2603.03291) that evaluates and
mitigates spurious biases in language reward models using **null-space
projection**.

Biases evaluated:
- **Position** — preference for answer slots (A/B/C/D) in multiple-choice
- **Sycophancy** — agreement with the user's stated opinion
- **Length** — preference for longer responses
- **Uncertainty** — penalizing hedged / uncertain language

Core method:
1. Build a **probe direction** from contrastive pairs (e.g. identical content at
   position A vs B).
2. **Project out** that direction from the model's hidden states (null-space
   projection).
3. Evaluate whether the targeted bias drops **without** harming reward accuracy.

### Important: this is NOT fine-tuning
The pipeline is **inference + hidden-state extraction + linear algebra**. No
model weights are trained. Any execution backend must provide:
1. **Reward scores** from a reward model.
2. **Per-layer hidden states / residual-stream activations** (the method reads
   and edits these).
3. Token-level log-probs / perplexity for `eval_perplexity.py`.

Do not introduce a LoRA/QLoRA/fine-tuning path — it is out of scope.

## Repository layout (verify against the tree)

```
configs/                    # YAML configs, symlinked into experiments/ and notebooks/
├── default_values.yaml     # defaults for all runs
├── models.yaml             # model + dataset registry
├── {length,position,sycophancy,uncertainty,rewardbench}_*.yaml
src/nb/
├── datasets/               # dataset loading & formatting
├── experiments/            # experiment orchestration
└── nullbias/               # probe building & null-space projection (backend-agnostic math)
experiments/                # CLI entry points
├── run_experiment.py
├── run_all.py
├── run_rewardbench_multiprobe.py
├── eval_perplexity.py
└── eval_rb2_data.py
notebooks/                  # run_experiment.ipynb, run_rewardbench_multiprobe.ipynb (expose run())
tests/
requirements.txt
```

## Config precedence
`configs/default_values.yaml` (defaults) < per-experiment YAML < keyword / CLI overrides.

## Running

```bash
# single experiment
python experiments/run_experiment.py --config configs/position_skywork_gsm8k.yaml

# CLI overrides
python experiments/run_experiment.py \
    --bias-type position \
    --model-path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --dataset-source guipenedo/gsm8k-mc

# all experiments (optionally filtered)
python experiments/run_all.py --filter position
python experiments/run_all.py --list

# RB2 multi-probe evaluation
python experiments/run_rewardbench_multiprobe.py --config configs/rewardbench_skywork.yaml

# analysis (no GPU)
python experiments/eval_perplexity.py
python experiments/eval_rb2_data.py
```

## Hardware / device (current state)
- `--device` flag: `cuda` (default; multi-GPU via HF `accelerate` `device_map="auto"`),
  `cuda:N`, `cpu`, `auto`, **`mlx`** (Apple Silicon).
- `auto`: CUDA host → `cuda` (unchanged); Apple Silicon w/o CUDA → `mlx` when `mlx-lm`
  installed, else `cpu`.
- MPS (PyTorch Metal) is **not** used; the Apple Silicon path is the MLX backend.
- Primary models are Llama-based reward models (e.g. `Skywork-Reward-Llama-3.1-8B-v0.2`).
- There is also a DeBERTa-based perplexity/RM eval (`perplexity_eval_RM_specialDeberta.py`)
  with `auto`→`cpu` fallback and `bfloat16`→`float32` downgrade. **DeBERTa is CPU-only on
  Mac** (no MLX port — encoder-only, unsupported by mlx-lm).
- Install flavors: `requirements.txt` (cross-platform base), `requirements-mlx.txt`
  (`mlx`, `mlx-lm`; prefer a Python 3.12/3.13 venv — 3.14 MLX wheels may be missing),
  `requirements-cuda.txt` (`vllm`).

## MLX backend (Apple Silicon) — implemented
Target hardware: 14" MacBook Pro, **base Apple M4, 32 GB** unified memory. MLX is an
**additive, opt-in dev/prototyping** path; CUDA/CPU paths are unchanged.

Architecture (`src/nb/backends/`):
- `base.py` — `ModelBackend` ABC: `embed_texts`, `score_texts`, `score_texts_both`,
  all returning **CPU float32 torch tensors**.
- `__init__.py` — `select_backend(device)` (routing) + `create_backend(config)`
  (factory). The transformers path returns the **raw HF model** (unchanged); only
  MLX is a `ModelBackend`.
- `mlx_backend.py` — loads the transformer **backbone** via `mlx-lm`'s model registry
  (keyed by `config.json` `model_type`; Llama/Qwen supported) from the checkpoint's
  own safetensors, and loads the scalar reward **`score` head** separately into a
  torch `nn.Linear`. Forward = `backbone(input_ids)` (post-final-norm hidden states),
  then the **score head + null-space projection run in shared CPU-torch** → numerical
  drift vs. the reference is confined to the backbone alone.

Integration: `nullbias/probe.py`'s `get_embeddings` / `get_rewards_with_nulling` /
`get_rewards_both` **dispatch** to a backend when their `model` arg is a `ModelBackend`;
`BiasExperiment.load_model` sets `self.model` to the backend. So every experiment routes
through MLX with no per-call-site changes, and the pure math (`project_to_null_space`,
`gram_schmidt`, diff-of-means, metrics) is untouched.

Key facts:
- Only `hidden_states[-1]` is used → `mlx-lm` `model.model(input_ids)` suffices (no deep
  instrumentation needed).
- Batched scoring uses **right padding** so causal masking makes the last non-pad token
  independent of trailing pads (matches the transformers reference).
- **bf16 default** (matches CUDA bf16 → best parity). `--mlx-quant 4bit/8bit` is opt-in,
  **not for publishable numbers**.
- Parity gate (`experiments/parity_check.py`): MLX-bf16 vs CPU-fp32 → reward Pearson
  r ≥ 0.99, **significant** probe-subspace overlap ≥ 0.99 (mean of principal-angle
  cosines over non-degenerate directions; multi-vector position bases have one
  degenerate orthonormalized direction that is precision-noise in both runs and is
  excluded), and **headline** bias-metric |Δ| ≤ 0.02 (overall accuracy / bias
  magnitude / gaps — per-bucket sub-stats like `accuracy_when_A`, `position_C_pct`
  are reported as diagnostics, not gated, since they quantize at 1/N_bucket).
- Parity configs (public 0.6B; the existing `*_shp_qwen3_*` configs point at a
  non-existent local cluster path, not a HF id):
  `configs/position_skywork_qwen06_gsm8k.yaml` (multi-vector probe) and
  `configs/uncertainty_skywork_qwen06_gsm8k.yaml` (single-vector probe).
- **Observed parity (Skywork-Qwen3-0.6B, n=100):** reward r ≈ 0.999, probe overlap
  ≈ 0.995 (position) / 0.9998 (uncertainty), baseline metrics |Δ| ≈ 0.01. **Caveat:**
  *nulled/debiased* headline metrics can differ by ~0.03 (≈3 comparison-flips/100) —
  a stable **bf16-vs-fp32 precision effect near decision ties**, not a framework bug
  (reward correlation is 0.999). Use fp32/CUDA for publishable *debiased* numbers.
- 8B Llama (`Skywork-Reward-V2-Llama-3.1-8B`) runs on MLX (bf16 ≈ 16 GB). Its fp32
  parity *reference* does not fit in 32 GB RAM — run that reference on CUDA (or use
  `parity_check.py --ref-dtype bfloat16` for a same-precision framework check).

## Conventions
- Config-driven; do not hardcode model/dataset paths — use `configs/models.yaml`.
- Keep changes minimal and reviewable; preserve the existing CLI and notebook
  `run()` interfaces.
- Add tests for new backend code, including an MLX smoke test that runs a tiny
  experiment end-to-end.