# Mechanistic Reward Shaping

Evaluates and mitigates spurious biases in reward models using null-space projection.

## Biases Evaluated

- **Position**: Preference for answer positions (A/B/C/D) in MCQ
  - **Multi-class variant** (experimental): Variable-class position bias with custom labels (2-4 classes)
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

# Multi-class position bias (custom class labels, 2-4 classes)
...
run("experiments/position_multi_class_example.yaml")

# run_rewardbench_multiprobe.ipynb — multi-probe RB2 evaluation
...
run("configs/rewardbench_skywork.yaml")
...
run("configs/rewardbench_skywork.yaml", null_alpha=0.5)
```

### Multi-Class Position Bias

For custom class labels and variable position counts (2-4 classes), use the `position_multi_class` dataset:

```bash
# CLI
python experiments/run_experiment.py --config experiments/position_multi_class_example.yaml

# Or with custom labels at runtime
python experiments/run_experiment.py \
    --bias-type position \
    --model Skywork/Skywork-Reward-Llama-3.1-8B \
    --dataset-source guipenedo/gsm8k-mc \
    --extra-dataset_class position_multi_class \
    --extra-num_classes 3
```

See [MULTI_CLASS_USAGE.md](MULTI_CLASS_USAGE.md) for detailed documentation and examples.

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

## Dataset Formats

This codebase supports multiple input formats for datasets:

- **Standard MCQ** (4-choice): GSM8K-MC, MMLU, PlausibleQA
- **Multi-class MCQ** (2-4 choice): Custom variable-class position bias experiments
- **Freeform**: BigBench and generative task formats

See [DATASET_FORMATS.md](DATASET_FORMATS.md) for detailed documentation on all supported input schemas and how to work with custom datasets.

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

```bash
pip install -r requirements.txt
```
