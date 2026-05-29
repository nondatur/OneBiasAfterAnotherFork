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
    --bias_type position \
    --model_path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --dataset_source guipenedo/gsm8k-mc

# Run all experiments (optionally filtered)
python experiments/run_all.py --filter position
python experiments/run_all.py --list            # preview without running

# Multi-probe RewardBench 2 evaluation
python experiments/run_rewardbench_multiprobe.py --config configs/rewardbench_skywork.yaml

# Analysis scripts (no GPU required)
uv run experiments/eval_perplexity.py     # reward–perplexity correlation
uv run experiments/eval_rb2_data.py       # RB2 probe-debiasing analysis
```

### Notebooks (Python API)

Each notebook exposes a `run()` function driven by YAML configs and keyword overrides.
Defaults are read from `configs/default_values.yaml`; per-experiment YAMLs override
them; keyword arguments override everything.

```python
# run_experiment.ipynb — single bias experiment
run("configs/length_skywork.yaml")
run("configs/position_skywork_gsm8k.yaml", device="cpu", batch_size=4)

# run_all.ipynb — full sweep
run_all()
run_all(filter_str="skywork", device="cuda")
run_all(list_only=True)                        # preview without running

# run_rewardbench_multiprobe.ipynb — multi-probe RB2 evaluation
run("configs/rewardbench_skywork.yaml")
run("configs/rewardbench_skywork.yaml", null_alpha=0.5)

# eval_perplexity.ipynb — reward–perplexity correlation (no GPU)
run()
run(corr_kind="pearson", nll_norm="token")

# eval_rb2_data.ipynb — RB2 OOD analysis (no GPU)
run()
run(filepath="rb2_data/my_results.json")
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
#├── run_all.ipynb
└── run_rewardbench_multiprobe.ipynb
#├── eval_perplexity.ipynb
#└── eval_rb2_data.ipynb
```

## Hardware & Device Support

All experiment scripts accept a `--device` flag that controls where models run.

| Value | Behaviour |
|---|---|
| `cuda` *(default)* | Multi-GPU — `device_map="auto"` shards the model across all visible GPUs |
| `cuda:0`, `cuda:1`, … | Single specified GPU |
| `cpu` | CPU-only inference (slow; useful for small models or debugging) |
| `auto` | Same as `cuda` for model loading; `perplexity_eval_RM_specialDeberta.py` also uses this as its default and falls back to `cpu` automatically if no GPU is present |

```bash
# Multi-GPU (default)
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

**Not supported:** Apple MPS (Metal). The codebase targets CUDA-capable hardware only.

## Requirements

```bash
pip install -r requirements.txt
```
