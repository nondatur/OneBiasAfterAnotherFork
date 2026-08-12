# Results snapshot: demographic reward-model bias

Metric result files (JSON) from the demographic protected-attribute bias experiments, all on
**Skywork-Reward-V2-Qwen3-0.6B**. These are the numbers behind the write-ups; they contain metrics,
configs, and record IDs only. **No raw datasets, essay text, or scored inputs are included here.**

## Scope of this snapshot
Everything here is **one reward model (0.6B), one seed (`42`), and n between 50 and 300 per cell.** No
confidence intervals or significance tests are computed, so small differences between cells should not be
read as real. The `auto_influence` statistic is a preference *rate* and saturates at 1.0, which makes the
cells at ceiling uninformative about effect *size*; where magnitude matters, prefer the standpoint arm's
`identity_gap` / `main_effect`, which are in raw reward units against a no-standpoint baseline. The
scaling study across the larger reward models is not represented here.

## Substrate change (2026-08): the hiring domain moved to real data
Every `*_cv_*` result file here was produced on the **synthetic** CV substrate
(`cv_ingest.generate_candidates`, deterministic at seed 42). As of 2026-08 the `cv` domain loads
**real biographies** (Bias-in-Bios) instead, so these files are *not* reproducible from the current
`domain=cv` — regenerate them via `experiments/generate_cv_data.py`, which is retained for exactly
that purpose. The erasure, reasoning and decision-response arms have not yet been re-run on the real
substrate.

## Why the data and scored inputs are not in the repo
- **Size:** the raw corpora are ~660 MB.
- **Licensing / redistribution:** PERSUADE 2.0 is **CC BY-NC-SA 4.0** (no redistribution of derived
  essays), ASAP-AES is under the **Kaggle 2012 competition terms**, and German Credit is CC-BY-4.0.
- **Privacy:** Bias-in-Bios (MIT) is ~400k biographies of **identifiable real people** scraped from
  Common Crawl. Neither the corpus nor the derived pairs are committed; only aggregate metrics leave
  the machine, and the dataset's real gender label is never rendered into any scored text.
  The matched-pair datasets embed derived essay/profile text, so they are **regenerated locally** from
  user-downloaded corpora via the generators in `experiments/` (e.g. `generate_edu_data.py`,
  `generate_positioned_data.py`), and the per-example scored inputs (`artifacts/raw_data/`) are excluded
  for the same reason. These result JSONs carry only aggregate metrics and are safe to share.

## What each file family is
- `battery_*` — auto-influence robustness battery (per axis x encoding, per-template, alpha-sweep).
- `crossinf_*` — cross-influence (reliability-harm) with `acc_baseline` / `baseline_tracks_quality`.
- `demographic_*_results` — single-axis auto-influence (baseline vs null-space-projected).
- `erasure_*` — LEACE + non-linear (MLP) probe recoverability (the low- vs high-complexity / TaCo test).
- `reasoning_*`, `decision_*` — the model-response verdict arms (reasoning-correctness 2x2, decision-response).
- `maineffect_edupos_*`, `battery_edupos_*` — positioned-argument (A2) standpoint-credibility arm
  (`maineffect_*` = neutral-baseline decomposition into main effect vs identity gap; `_positions_`,
  `_paraphrase_`, `_stance_` are the robustness runs).

Regenerate any of these end to end with the corresponding `experiments/run_*.py` after downloading the
source corpora (see `data/demographic/education/README.md`).

## Probe directions (`probes/`)
The difference-of-means bias directions used for null-space projection: one `probe.pt` (a single
`[hidden_dim]` float tensor) plus `metadata.json` (norms, separation, probe accuracy) per axis. Safe to
share (a single direction vector, no text, derived from open-weights activations) and enough to apply or
verify the null-space projection without re-extracting activations. Partial set: only runs with `save_probe`
enabled (credit and CV, sex and intersection) are saved; all other probes are rebuilt in-memory by the
runners and are regenerable.
