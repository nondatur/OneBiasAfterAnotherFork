# Mechanistic Reward Shaping

Evaluates and mitigates spurious biases in reward models using null-space projection.

## Biases Evaluated

- **Position**: Preference for answer positions (A/B/C/D) in MCQ
- **Sycophancy**: Agreement with user's stated opinion
- **Length**: Preference for longer responses
- **Uncertainty**: Penalizing hedged/uncertain language

## Method

1. Build a **probe direction** from contrastive pairs (e.g., same content at position A vs B)
2. **Project out** the probe direction from hidden states via null-space projection
3. Evaluate whether bias is reduced without harming accuracy

## Usage

### Scripts (CLI)

```bash
# Run a single experiment from a config file
python experiments/run_experiment.py --config configs/position_skywork_gsm8k.yaml

# Run with CLI overrides
python experiments/run_experiment.py \
    --bias-type position \
    --model-path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --dataset-source guipenedo/gsm8k-mc

# Run all experiments (optionally filtered)
python experiments/run_all.py --filter position
python experiments/run_all.py --list            # preview without running

# Multi-probe RewardBench 2 evaluation
python experiments/run_rewardbench_multiprobe.py --config configs/rewardbench_skywork.yaml

# Analysis scripts (no GPU required)
python experiments/eval_perplexity.py     # reward–perplexity correlation
python experiments/eval_rb2_data.py       # RB2 probe-debiasing analysis
```

### Notebooks (Python API)

Each notebook exposes a `run()` function driven by YAML configs and keyword overrides.
Defaults are read from `configs/default_values.yaml`; per-experiment YAMLs override
them; keyword arguments override everything.

```python
# run_experiment.ipynb — single bias experiment
...
run("configs/length_skywork.yaml")
...
run("configs/position_skywork_gsm8k.yaml", device="cpu", batch_size=4)

# run_rewardbench_multiprobe.ipynb — multi-probe RB2 evaluation
...
run("configs/rewardbench_skywork.yaml")
...
run("configs/rewardbench_skywork.yaml", null_alpha=0.5)
```

## Notebooks

| Notebook | Description |
|---|---|
| [`run_experiment.ipynb`](notebooks/run_experiment.ipynb) | Run a single bias experiment end-to-end |
| [`run_rewardbench_multiprobe.ipynb`](notebooks/run_rewardbench_multiprobe.ipynb) | Multi-probe RB2 evaluation with combined null-space debiasing |

## Structure

```
configs/                    # Shared config directory (symlinked from experiments/ and notebooks/)
├── default_values.yaml     # Default parameter values for all notebooks
├── models.yaml             # Shared model & dataset registry
├── length_*.yaml     # Per-experiment YAML configs
├── position_*.yaml
├── sycophancy_*.yaml
├── uncertainty_*.yaml
└── rewardbench_*.yaml

src/nb/
├── datasets/               # Dataset loading & formatting
├── experiments/            # Experiment orchestration
└── nullbias/               # Probe building & projection

experiments/
├── configs -> ../configs   # Symlink to shared configs/
├── run_experiment.py
├── run_all.py
├── run_rewardbench_multiprobe.py
├── eval_perplexity.py
└── eval_rb2_data.py

notebooks/
├── configs -> ../configs   # Symlink to shared configs/
├── run_experiment.ipynb
└── run_rewardbench_multiprobe.ipynb
```

## Hardware & Device Support

All experiment scripts accept a `--device` flag that controls where models run.

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Adapts to the host: CUDA if present → same as `cuda` (multi-GPU, unchanged); else MLX on Apple Silicon when `mlx-lm` is installed; else CPU. `perplexity_eval_RM_specialDeberta.py` also uses `auto` and falls back to `cpu`. |
| `cuda` | Multi-GPU — `device_map="auto"` shards the model across all visible GPUs |
| `cuda:0`, `cuda:1`, … | Single specified GPU |
| `cpu` | CPU-only inference (slow; useful for small models or debugging) |
| `mlx` | **Apple Silicon (MLX) backend.** Runs the transformer backbone via `mlx`/`mlx-lm`; the reward `score` head and null-space projection run in shared CPU-torch. bf16 by default; add `--mlx-quant 4bit`/`8bit` for memory headroom (opt-in, not for publishable numbers). Requires `pip install -r requirements-mlx.txt`. |

```bash
# Default: auto (multi-GPU on a CUDA host; MLX on Apple Silicon; else CPU)
python experiments/run_experiment.py --config configs/position_skywork_gsm8k.yaml

# Single GPU
python experiments/run_experiment.py --config configs/position_skywork_gsm8k.yaml --device cuda:1

# CPU only
python experiments/run_experiment.py --config configs/position_skywork_gsm8k.yaml --device cpu
```

**Multi-GPU behaviour:** when `--device cuda` (or `auto`) is used, HuggingFace `accelerate`'s
`device_map="auto"` is applied. This distributes model layers evenly across all GPUs visible
to the process. No manual `CUDA_VISIBLE_DEVICES` filtering is needed; the number of GPUs used
is determined automatically at runtime.

**CPU behaviour:** all scripts work on CPU. Expect significantly slower throughput.
`bfloat16` is automatically downgraded to `float32` where needed
(`perplexity_eval_RM_specialDeberta.py`).

**Apple Silicon:** use `--device mlx` (the MLX backend), not PyTorch MPS. MPS is not
used. The MLX path is a **development/prototyping** target — validate publishable
numbers against full-precision CUDA (see `experiments/parity_check.py`).

## References

When using this codebase, please cite:

- https://arxiv.org/abs/2603.03291

```bibtex
@misc{fein2026biasanothermechanisticreward,
      title={One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models}, 
      author={Daniel Fein and Max Lamparth and Violet Xiang and Mykel J. Kochenderfer and Nick Haber},
      year={2026},
      eprint={2603.03291},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.03291}, 
}
```

## Requirements

The base requirements install on both Linux/CUDA and macOS/Apple Silicon:

```bash
pip install -r requirements.txt
```

Backend-specific extras:

```bash
# Apple Silicon (MLX) — prefer a Python 3.12/3.13 venv (3.14 MLX wheels may be missing)
pip install -r requirements-mlx.txt

# NVIDIA CUDA extras (vLLM for fast GSM8K generation)
pip install -r requirements-cuda.txt
```

Numerical parity (Apple Silicon MLX vs. CPU/transformers reference):

```bash
python experiments/parity_check.py \
    --config configs/position_skywork_qwen06_gsm8k.yaml --max-examples 50 --probe-size 100
```
