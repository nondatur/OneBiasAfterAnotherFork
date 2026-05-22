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

```bash
# Run a single experiment from a config file
python experiments/run_experiment.py --config experiments/configs/position_skywork_gsm8k.yaml

# Run with CLI overrides
python experiments/run_experiment.py \
    --bias_type position \
    --model_path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --dataset_source guipenedo/gsm8k-mc

# Run all experiments (optionally filtered)
python experiments/run_all.py --filter position
python experiments/run_all.py --list            # preview without running

# Multi-probe RewardBench 2 evaluation
python experiments/run_rewardbench_multiprobe.py \
    --model Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --probes-dir artifacts/probes

# Analysis scripts (no GPU required)
uv run experiments/eval_perplexity.py     # reward–perplexity correlation
uv run experiments/eval_rb2_data.py       # RB2 probe-debiasing analysis
```

## Notebooks

Interactive versions of all experiment scripts are available in `notebooks/`.
Each notebook mirrors its corresponding script with one cell per function and a
short markdown introduction.

| Notebook | Description |
|---|---|
| [`eval_perplexity.ipynb`](notebooks/eval_perplexity.ipynb) | Panel-relative log-prob vs reward-model correlation analysis |
| [`eval_rb2_data.ipynb`](notebooks/eval_rb2_data.ipynb) | RewardBench 2 OOD evaluation with/without probe debiasing |
| [`run_all.ipynb`](notebooks/run_all.ipynb) | Orchestrate a full experiment sweep across all models & datasets |
| [`run_experiment.ipynb`](notebooks/run_experiment.ipynb) | Run a single bias experiment end-to-end |
| [`run_rewardbench_multiprobe.ipynb`](notebooks/run_rewardbench_multiprobe.ipynb) | Multi-probe RB2 evaluation with combined null-space debiasing |

Experiment YAML configs and `models.yaml` are mirrored under `notebooks/configs/`.

## Structure

```
src/nb/
├── datasets/       # Dataset loading & formatting
├── experiments/    # Experiment orchestration
└── nullbias/       # Probe building & projection

experiments/
├── configs/        # YAML experiment configs
├── models.yaml     # Shared model & dataset registry
├── run_experiment.py
├── run_all.py
├── run_rewardbench_multiprobe.py
├── eval_perplexity.py
└── eval_rb2_data.py

notebooks/
├── configs/        # Mirror of experiments/configs/
├── models.yaml     # Mirror of experiments/models.yaml
├── eval_perplexity.ipynb
├── eval_rb2_data.ipynb
├── run_all.ipynb
├── run_experiment.ipynb
└── run_rewardbench_multiprobe.ipynb
```

## Requirements

```bash
pip install -r requirements.txt
```
