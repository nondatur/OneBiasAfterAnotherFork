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
  `cuda:N`, `cpu`, `auto`.
- **CUDA-only.** Apple MPS (Metal) is explicitly NOT supported.
- Primary models are Llama-based reward models (e.g. `Skywork-Reward-Llama-3.1-8B-v0.2`).
- There is also a DeBERTa-based perplexity/RM eval (`perplexity_eval_RM_specialDeberta.py`)
  with `auto`→`cpu` fallback and `bfloat16`→`float32` downgrade.

## Active goal: add a local MLX backend (Apple Silicon)
Target hardware: 14" MacBook Pro, **base Apple M4, 32 GB** unified memory.

- Add an MLX execution path as an **additive, opt-in** backend (e.g. `--device mlx`,
  plus auto-detect on Apple Silicon). Do not break or regress the CUDA/CPU paths.
- Keep `nullbias/` math and experiment orchestration **backend-agnostic**; isolate
  framework specifics behind a backend interface.
- The MLX path must expose **reward scores AND per-layer hidden states**. `mlx-lm`
  does not expose hidden states or reward heads out of the box — expect custom
  forward instrumentation and custom head loading.
- Prefer quantized (4-bit/8-bit) MLX weights for memory, **but**: quantization can
  shift bias measurements. Treat the local MLX path as a **development/prototyping**
  target and validate research-grade numbers against full-precision CUDA. Include a
  numerical-parity check on a small sample.

## Conventions
- Config-driven; do not hardcode model/dataset paths — use `configs/models.yaml`.
- Keep changes minimal and reviewable; preserve the existing CLI and notebook
  `run()` interfaces.
- Add tests for new backend code, including an MLX smoke test that runs a tiny
  experiment end-to-end.