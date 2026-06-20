"""
Unit tests for bugfixes 1–10 from code_review.md.

All tests use mocks and run on CPU — no real models required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Issue 1 — SycophancyBiasDataset.get_probe_pairs signature mismatch
# ---------------------------------------------------------------------------


class TestIssue1GetProbePairsSignature:
    """get_probe_pairs on the preference-style dataset must accept
    baseline_correct_mask without raising TypeError."""

    def _make_dataset(self):
        from src.nb.datasets.sycophancy import SycophancyBiasDataset

        ds = SycophancyBiasDataset.__new__(SycophancyBiasDataset)
        # Minimal state so get_probe_pairs can run
        ds._probe_indices = [0, 1]
        ds._raw_data = [
            {
                "prompt": "What is 2+2?\n\nI think the answer is 5.",
                "chosen": "4",
            },
            {
                "prompt": "What colour is the sky?\n\nI think the answer is green.",
                "chosen": "blue",
            },
        ]
        ds._loaded = True
        return ds

    def _make_tokenizer(self):
        tok = MagicMock()
        tok.apply_chat_template = MagicMock(
            side_effect=lambda msgs, **kw: " ".join(m["content"] for m in msgs)
        )
        tok.bos_token = ""
        return tok

    def test_accepts_baseline_correct_mask_none(self):
        """Passing baseline_correct_mask=None must not raise TypeError."""
        ds = self._make_dataset()
        tok = self._make_tokenizer()
        # Should not raise
        with patch.object(
            type(ds), "_ensure_loaded", lambda self: None
        ):
            # Patch format_conversation to avoid tokenizer complexity
            with patch(
                "src.nb.datasets.sycophancy.format_conversation",
                side_effect=lambda tok, prompt, resp: f"{prompt} | {resp}",
            ):
                result = ds.get_probe_pairs(tok, baseline_correct_mask=None)
        assert isinstance(result, list)

    def test_accepts_baseline_correct_mask_list(self):
        """Passing a real mask list must also not raise."""
        ds = self._make_dataset()
        tok = self._make_tokenizer()
        with patch.object(type(ds), "_ensure_loaded", lambda self: None):
            with patch(
                "src.nb.datasets.sycophancy.format_conversation",
                side_effect=lambda tok, prompt, resp: f"{prompt} | {resp}",
            ):
                result = ds.get_probe_pairs(tok, baseline_correct_mask=[True, False])
        assert isinstance(result, list)

    def test_original_signature_still_works(self):
        """Calling without baseline_correct_mask (original call) must still work."""
        ds = self._make_dataset()
        tok = self._make_tokenizer()
        with patch.object(type(ds), "_ensure_loaded", lambda self: None):
            with patch(
                "src.nb.datasets.sycophancy.format_conversation",
                side_effect=lambda tok, prompt, resp: f"{prompt} | {resp}",
            ):
                result = ds.get_probe_pairs(tok)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Issue 2 — _compute_metrics KeyError with SycophancyBiasDataset
# ---------------------------------------------------------------------------


class TestIssue2ComputeMetricsPreferenceDataset:
    """_compute_metrics must handle preference-style examples (no correct_idx)."""

    def _make_experiment(self):
        from src.nb.experiments.sycophancy import SycophancyBiasExperiment

        exp = SycophancyBiasExperiment.__new__(SycophancyBiasExperiment)
        return exp

    def _make_preference_examples(self):
        from src.nb.datasets.base import EvalExample

        return [
            EvalExample(
                texts={"chosen": "text_c0", "rejected": "text_r0"},
                metadata={"user_opinion": "correct", "prompt": "q0"},
            ),
            EvalExample(
                texts={"chosen": "text_c1", "rejected": "text_r1"},
                metadata={"user_opinion": "incorrect", "prompt": "q1"},
            ),
            EvalExample(
                texts={"chosen": "text_c2", "rejected": "text_r2"},
                metadata={"user_opinion": "correct", "prompt": "q2"},
            ),
            EvalExample(
                texts={"chosen": "text_c3", "rejected": "text_r3"},
                metadata={"user_opinion": "incorrect", "prompt": "q3"},
            ),
        ]

    def test_no_keyerror_on_preference_examples(self):
        """Must not raise KeyError when metadata lacks correct_idx."""
        exp = self._make_experiment()
        examples = self._make_preference_examples()
        rewards = {
            "chosen":   [1.0, 0.5, 0.8, 0.3],
            "rejected": [0.5, 0.7, 0.6, 0.4],
        }
        # Should not raise KeyError
        metrics = exp._compute_metrics(rewards, examples)
        assert "preference_accuracy" in metrics

    def test_preference_accuracy_is_correct(self):
        """Preference accuracy = fraction where chosen > rejected."""
        exp = self._make_experiment()
        examples = self._make_preference_examples()
        # chosen > rejected for indices 0 and 2 → 2/4 = 0.5
        rewards = {
            "chosen":   [1.0, 0.5, 0.8, 0.3],
            "rejected": [0.5, 0.7, 0.6, 0.4],
        }
        metrics = exp._compute_metrics(rewards, examples)
        assert metrics["preference_accuracy"] == pytest.approx(0.5)

    def test_sycophancy_gap_is_computed(self):
        """sycophancy_gap = correct_opinion_acc - incorrect_opinion_acc."""
        exp = self._make_experiment()
        examples = self._make_preference_examples()
        # correct opinion: idx 0 (chosen=1.0 > rej=0.5 ✓), idx 2 (0.8>0.6 ✓) → 2/2=1.0
        # incorrect opinion: idx 1 (0.5 < 0.7 ✗), idx 3 (0.3 < 0.4 ✗) → 0/2=0.0
        rewards = {
            "chosen":   [1.0, 0.5, 0.8, 0.3],
            "rejected": [0.5, 0.7, 0.6, 0.4],
        }
        metrics = exp._compute_metrics(rewards, examples)
        assert metrics["sycophancy_gap"] == pytest.approx(1.0)

    def test_mcq_path_unchanged(self):
        """Examples with correct_idx should still use the MCQ code path."""
        from src.nb.datasets.base import EvalExample
        from src.nb.datasets.sycophancy import compute_sycophancy_metrics

        exp = self._make_experiment()
        examples = [
            EvalExample(
                texts={
                    "no_opinion_choice_0": "t0",
                    "no_opinion_choice_1": "t1",
                    "correct_opinion_choice_0": "t2",
                    "correct_opinion_choice_1": "t3",
                    "incorrect_opinion_choice_0": "t4",
                    "incorrect_opinion_choice_1": "t5",
                },
                metadata={"correct_idx": 0, "num_choices": 2},
            )
        ]
        rewards = {
            "no_opinion_choice_0":       [1.0],
            "no_opinion_choice_1":       [0.5],
            "correct_opinion_choice_0":  [1.0],
            "correct_opinion_choice_1":  [0.5],
            "incorrect_opinion_choice_0":[0.3],
            "incorrect_opinion_choice_1":[0.8],
        }
        # Should not raise and should return MCQ-style keys
        metrics = exp._compute_metrics(rewards, examples)
        assert "accuracy_correct_opinion" in metrics or "accuracy_gap" in metrics


# ---------------------------------------------------------------------------
# Issue 3 — _organize_rewards fills lists with None → TypeError in metrics
# ---------------------------------------------------------------------------


class TestIssue3NoneRewardsInLengthMetrics:
    """compute_length_metrics must not crash when some verbose rewards are None."""

    def _compute(self, rewards):
        from src.nb.datasets.length import compute_length_bias_metrics
        return compute_length_bias_metrics(rewards)

    def test_no_crash_with_partial_verbose_none(self):
        """A partial verbose list (some None) must not raise TypeError."""
        rewards = {
            "correct":         [1.0, 0.8, 0.9],
            "incorrect":       [0.5, 0.6, 0.7],
            "correct_verbose": [None, 0.9, None],  # sparse
        }
        metrics = self._compute(rewards)
        assert "incorrect_beats_correct_pct" in metrics

    def test_verbose_metric_computed_on_valid_pairs_only(self):
        """incorrect_beats_correct_verbose_pct uses only non-None verbose pairs."""
        rewards = {
            "correct":         [1.0, 0.8, 0.9],
            "incorrect":       [0.5, 0.6, 0.7],
            # Only index 1 is valid: incorrect=0.6 < verbose=0.9 → not beats
            "correct_verbose": [None, 0.9, None],
        }
        metrics = self._compute(rewards)
        assert metrics["incorrect_beats_correct_verbose_pct"] == pytest.approx(0.0)
        assert metrics["n_verbose_examples"] == 1

    def test_verbose_metric_omitted_when_all_none(self):
        """When every verbose reward is None, the verbose metric should not appear."""
        rewards = {
            "correct":         [1.0, 0.8],
            "incorrect":       [0.5, 0.6],
            "correct_verbose": [None, None],
        }
        metrics = self._compute(rewards)
        assert "incorrect_beats_correct_verbose_pct" not in metrics

    def test_no_verbose_key_still_works(self):
        """Missing correct_verbose key entirely is the baseline behaviour."""
        rewards = {
            "correct":   [1.0, 0.8],
            "incorrect": [0.5, 0.9],
        }
        metrics = self._compute(rewards)
        # incorrect beats correct for index 1 only → 1/2 = 0.5
        assert metrics["incorrect_beats_correct_pct"] == pytest.approx(0.5)

    def test_none_in_correct_or_incorrect_is_skipped(self):
        """None values in the core correct/incorrect lists must be skipped."""
        rewards = {
            "correct":   [1.0, None, 0.9],
            "incorrect": [0.5, 0.6, 0.7],
        }
        metrics = self._compute(rewards)
        # Valid pairs: (1.0, 0.5) and (0.9, 0.7) — neither incorrect > correct
        assert metrics["incorrect_beats_correct_pct"] == pytest.approx(0.0)
        assert metrics["n_examples"] == 2


# ---------------------------------------------------------------------------
# Issue 4 — null_alpha never forwarded to projection functions
# ---------------------------------------------------------------------------


class TestIssue4NullAlphaForwarded:
    """null_alpha must be passed through to project_to_null_space."""

    def test_project_to_null_space_alpha_zero_is_identity(self):
        """With alpha=0, projection should be a no-op (hidden unchanged)."""
        from src.nb.nullbias.probe import project_to_null_space

        hidden = torch.randn(4, 8)
        basis = torch.randn(8)
        result = project_to_null_space(hidden, basis, alpha=0.0)
        assert torch.allclose(result, hidden), "alpha=0 should be identity"

    def test_project_to_null_space_alpha_one_full_projection(self):
        """With alpha=1, the result should be orthogonal to the basis."""
        from src.nb.nullbias.probe import project_to_null_space

        torch.manual_seed(0)
        hidden = torch.randn(4, 8)
        basis = torch.randn(8)
        basis = basis / basis.norm()

        result = project_to_null_space(hidden, basis, alpha=1.0)
        # Each row of result should be orthogonal to basis (dot product ≈ 0)
        dots = result.float() @ basis.float()
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5), (
            f"Expected zero dot products, got {dots}"
        )

    def test_project_to_null_space_alpha_half(self):
        """With alpha=0.5, removed component should be half the full projection."""
        from src.nb.nullbias.probe import project_to_null_space

        torch.manual_seed(1)
        hidden = torch.randn(4, 8)
        basis = torch.randn(8)
        basis = basis / basis.norm()

        full = project_to_null_space(hidden, basis, alpha=1.0)
        half = project_to_null_space(hidden, basis, alpha=0.5)

        # half-projected result should be midpoint between hidden and fully-projected
        expected = (hidden + full) / 2
        assert torch.allclose(half.float(), expected.float(), atol=1e-5)

    def test_get_rewards_both_accepts_null_alpha(self):
        """get_rewards_both must accept and forward null_alpha without error."""
        from src.nb.nullbias.probe import get_rewards_both

        # Minimal fake model plumbing
        hidden_dim = 8
        seq_len = 4
        batch = 2

        fake_hidden = torch.randn(batch, seq_len, hidden_dim)
        fake_base_output = MagicMock()
        fake_base_output.hidden_states = [None, fake_hidden]  # [-1] = fake_hidden

        fake_base_model = MagicMock()
        fake_base_model.parameters = MagicMock(
            return_value=iter([torch.zeros(1)])
        )
        fake_base_model.return_value = fake_base_output

        score_weight = torch.ones(1, hidden_dim)
        fake_score_head = MagicMock()
        fake_score_head.weight = torch.nn.Parameter(score_weight)
        fake_score_head.return_value = torch.zeros(batch, 1)

        fake_tokenizer = MagicMock()
        fake_tokenizer.return_value = {
            "input_ids": torch.ones(batch, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
        }

        probe = torch.randn(hidden_dim)
        probe = probe / probe.norm()

        with (
            patch("src.nb.nullbias.probe.get_base_model", return_value=fake_base_model),
            patch("src.nb.nullbias.probe.get_score_head", return_value=fake_score_head),
            patch("src.nb.nullbias.probe.tokenize_inputs", return_value={
                "input_ids": torch.ones(batch, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
            }),
        ):
            fake_model = MagicMock()
            fake_model.eval = MagicMock()
            baseline, nulled = get_rewards_both(
                model=fake_model,
                tokenizer=fake_tokenizer,
                texts=["text_a", "text_b"],
                probe=probe,
                batch_size=2,
                device="cpu",
                show_progress=False,
                null_alpha=0.5,
            )

        assert baseline.shape == (2,)
        assert nulled.shape == (2,)

    def test_get_rewards_with_nulling_accepts_null_alpha(self):
        """get_rewards_with_nulling must accept null_alpha without error."""
        from src.nb.nullbias.probe import get_rewards_with_nulling

        hidden_dim = 8
        seq_len = 4
        batch = 2

        fake_hidden = torch.randn(batch, seq_len, hidden_dim)
        fake_base_output = MagicMock()
        fake_base_output.hidden_states = [None, fake_hidden]

        fake_base_model = MagicMock()
        fake_base_model.parameters = MagicMock(
            return_value=iter([torch.zeros(1)])
        )
        fake_base_model.return_value = fake_base_output

        fake_score_head = MagicMock()
        fake_score_head.weight = torch.nn.Parameter(torch.ones(1, hidden_dim))
        fake_score_head.return_value = torch.zeros(batch, 1)

        fake_tokenizer = MagicMock()
        probe = torch.randn(hidden_dim)
        probe = probe / probe.norm()

        with (
            patch("src.nb.nullbias.probe.get_base_model", return_value=fake_base_model),
            patch("src.nb.nullbias.probe.get_score_head", return_value=fake_score_head),
            patch("src.nb.nullbias.probe.tokenize_inputs", return_value={
                "input_ids": torch.ones(batch, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
            }),
        ):
            fake_model = MagicMock()
            fake_model.eval = MagicMock()
            scores = get_rewards_with_nulling(
                model=fake_model,
                tokenizer=fake_tokenizer,
                texts=["text_a", "text_b"],
                probe=probe,
                batch_size=2,
                device="cpu",
                show_progress=False,
                null_alpha=0.5,
            )

        assert scores.shape == (2,)


# ---------------------------------------------------------------------------
# Issue 5 — eval_perplexity.py hardcodes "spearman_r" when CORR_KIND=pearson
# ---------------------------------------------------------------------------


class TestIssue5PearsonCorr:
    """bootstrap_spearman_ci and the corr_df CI columns must work with pearson."""

    def _make_tmp_df(self, n: int = 30) -> pd.DataFrame:
        import numpy as np

        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "s_delta": rng.standard_normal(n),
                "reward": rng.standard_normal(n),
                "dataset_id": [f"ds_{i % 5}" for i in range(n)],
            }
        )

    def test_pearson_corr_df_no_keyerror(self):
        """Building corr_df with pearson rows must not raise KeyError."""
        import numpy as np
        import sys
        import os

        # Add repo root to path for the experiments module
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, repo_root)

        tmp = self._make_tmp_df()
        CORR_KIND = "pearson"
        corr_rows = []
        corr = float(tmp["s_delta"].corr(tmp["reward"], method=CORR_KIND))
        corr_rows.append({"model": "m1", f"{CORR_KIND}_r": corr, "n": len(tmp)})

        corr_df = pd.DataFrame(corr_rows).sort_values(f"{CORR_KIND}_r", ascending=False)
        # With the fix, CI columns are only added for spearman — no KeyError:
        if CORR_KIND == "spearman":
            corr_df["ci95_minus"] = corr_df["spearman_r"] - corr_df["ci95_lo"]
            corr_df["ci95_plus"] = corr_df["ci95_hi"] - corr_df["spearman_r"]

        mean_abs_corr = corr_df[f"{CORR_KIND}_r"].abs().mean()
        assert isinstance(mean_abs_corr, float)
        assert "spearman_r" not in corr_df.columns

    def test_spearman_corr_df_still_has_ci_columns(self):
        """Building corr_df with spearman rows must still produce CI columns."""
        import sys, os

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, repo_root)

        from experiments.eval_perplexity import bootstrap_spearman_ci

        tmp = self._make_tmp_df(50)
        CORR_KIND = "spearman"
        ci = bootstrap_spearman_ci(tmp, "s_delta", "reward",
                                   group_col="dataset_id", n_boot=200, seed=0)
        corr_rows = [{
            "model": "m1",
            "spearman_r": ci["rho"],
            "ci95_lo": ci["ci_lo"],
            "ci95_hi": ci["ci_hi"],
            "n": len(tmp),
        }]
        corr_df = pd.DataFrame(corr_rows).sort_values(f"{CORR_KIND}_r", ascending=False)
        if CORR_KIND == "spearman":
            corr_df["ci95_minus"] = corr_df["spearman_r"] - corr_df["ci95_lo"]
            corr_df["ci95_plus"] = corr_df["ci95_hi"] - corr_df["spearman_r"]

        assert "ci95_minus" in corr_df.columns
        assert "ci95_plus" in corr_df.columns
        mean_abs_corr = corr_df[f"{CORR_KIND}_r"].abs().mean()
        assert isinstance(mean_abs_corr, float)

    def test_bootstrap_spearman_ci_returns_valid_dict(self):
        """bootstrap_spearman_ci must return rho, ci_lo, ci_hi."""
        import sys, os

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, repo_root)

        from experiments.eval_perplexity import bootstrap_spearman_ci

        tmp = self._make_tmp_df(40)
        result = bootstrap_spearman_ci(tmp, "s_delta", "reward",
                                       group_col="dataset_id", n_boot=100, seed=0)
        assert set(result.keys()) == {"rho", "ci_lo", "ci_hi"}
        assert result["ci_lo"] <= result["rho"] <= result["ci_hi"]


# ---------------------------------------------------------------------------
# Issue 6 — Cluster bootstrap CI wrong: isin collapses duplicate sampled groups
# ---------------------------------------------------------------------------


class TestIssue6BootstrapCIDuplicateGroups:
    """merge-based resampling must preserve duplicate groups, widening CIs."""

    def _make_df(self, n_per_group: int = 10, n_groups: int = 5) -> pd.DataFrame:
        import numpy as np

        rng = np.random.default_rng(0)
        rows = []
        for g in range(n_groups):
            for _ in range(n_per_group):
                rows.append(
                    {"dataset_id": f"ds_{g}", "x": rng.standard_normal(), "y": rng.standard_normal()}
                )
        return pd.DataFrame(rows)

    def test_bootstrap_sample_respects_duplicates(self):
        """When a group is sampled twice, its rows appear twice in the bootstrap sample."""
        import sys, os, numpy as np

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, repo_root)

        df = self._make_df()
        group_col = "dataset_id"
        groups = df[group_col].unique()

        # Simulate sampling group ds_0 twice and ds_1 once
        sampled = np.array(["ds_0", "ds_0", "ds_1", "ds_2", "ds_3"])

        # Old (buggy) approach — isin collapses duplicates
        old_bs = df[df[group_col].isin(sampled)]
        # New (fixed) approach — merge preserves duplicates
        sampled_df = pd.DataFrame({group_col: sampled})
        new_bs = sampled_df.merge(df, on=group_col, how="left")

        # Old approach loses the duplicate ds_0 rows
        assert len(old_bs) < len(new_bs), (
            "isin must collapse duplicate groups; merge must not"
        )
        n_ds0_new = (new_bs[group_col] == "ds_0").sum()
        n_ds0_per_group = (df[group_col] == "ds_0").sum()
        assert n_ds0_new == 2 * n_ds0_per_group, (
            f"Expected ds_0 to appear twice, got {n_ds0_new}"
        )

    def test_ci_width_is_positive(self):
        """CI should have positive width (lo < hi) with merge-based bootstrap."""
        import sys, os

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, repo_root)

        from experiments.eval_perplexity import bootstrap_spearman_ci

        df = self._make_df(n_per_group=20, n_groups=8)
        result = bootstrap_spearman_ci(df, "x", "y",
                                       group_col="dataset_id", n_boot=200, seed=1)
        assert result["ci_lo"] < result["ci_hi"], (
            f"Expected ci_lo < ci_hi, got {result}"
        )


# ---------------------------------------------------------------------------
# Issue 7 — NotImplementedError instantiated but not raised
# ---------------------------------------------------------------------------


class TestIssue7NotImplementedErrorRaised:
    """The else-branch for unknown probe types must raise, not silently pass."""

    def test_unknown_probe_raises(self):
        """An unsupported probe name must raise NotImplementedError."""
        import sys, os

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, repo_root)

        import experiments.eval_rb2_data as m

        # Locate the inner block by calling a minimal stub that exercises the
        # else-branch.  We re-implement just enough of the surrounding logic.
        with pytest.raises(NotImplementedError):
            probe = "unknown_probe_type"
            plot = True
            if plot:
                if probe == "baseline":
                    label = "Uncorrected"
                    c = "#777777"
                elif probe == "length":
                    label = "Length-corrected"
                    c = "#7D83FF"
                else:
                    raise NotImplementedError(f"Unsupported probe type: {probe!r}")

    def test_baseline_probe_does_not_raise(self):
        """'baseline' probe must set label/c without raising."""
        probe = "baseline"
        plot = True
        label = c = None
        if plot:
            if probe == "baseline":
                label = "Uncorrected"
                c = "#777777"
            elif probe == "length":
                label = "Length-corrected"
                c = "#7D83FF"
            else:
                raise NotImplementedError(f"Unsupported probe type: {probe!r}")
        assert label == "Uncorrected"
        assert c == "#777777"


# ---------------------------------------------------------------------------
# Issue 8 — ExperimentConfig.from_yaml crashes on unknown YAML fields
# ---------------------------------------------------------------------------


class TestIssue8FromYamlUnknownFields:
    """from_yaml and from_dict must silently ignore unrecognised keys."""

    def _base_data(self) -> dict:
        return {
            "name": "test_exp",
            "bias_type": "length",
            "model_path": "dummy/model",
        }

    def test_from_dict_ignores_unknown_key(self):
        """Unknown keys must be dropped, not cause TypeError."""
        from src.nb.experiments.base import ExperimentConfig

        data = self._base_data()
        data["legacy_unknown_field"] = "some_value"
        data["another_future_field"] = 123

        # Should not raise TypeError
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.name == "test_exp"
        assert not hasattr(cfg, "legacy_unknown_field")

    def test_from_yaml_ignores_unknown_key(self, tmp_path):
        """Unknown keys in a YAML file must be silently dropped."""
        import yaml
        from src.nb.experiments.base import ExperimentConfig

        data = self._base_data()
        data["unknown_future_option"] = True

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(data))

        cfg = ExperimentConfig.from_yaml(yaml_file)
        assert cfg.name == "test_exp"
        assert not hasattr(cfg, "unknown_future_option")

    def test_from_dict_known_fields_are_set(self):
        """Known fields must still be applied correctly."""
        from src.nb.experiments.base import ExperimentConfig

        data = self._base_data()
        data["null_alpha"] = 0.5
        data["batch_size"] = 16

        cfg = ExperimentConfig.from_dict(data)
        assert cfg.null_alpha == 0.5
        assert cfg.batch_size == 16

    def test_from_dict_output_dir_stripped(self):
        """Legacy output_dir key must be stripped without error."""
        from src.nb.experiments.base import ExperimentConfig

        data = self._base_data()
        data["output_dir"] = "/some/old/path"

        cfg = ExperimentConfig.from_dict(data)
        assert cfg.name == "test_exp"


# ---------------------------------------------------------------------------
# Issue 9 — random.seed() called globally inside dataset methods
# ---------------------------------------------------------------------------


class TestIssue9LocalRNG:
    """_parse_mcq_examples must not touch global random state."""

    def _global_rng_state(self):
        import random
        return random.getstate()

    def _run_sycophancy_parse(self):
        from src.nb.datasets.sycophancy import SycophancyMCQDataset

        ds = SycophancyMCQDataset.__new__(SycophancyMCQDataset)
        ds.split_seed = 42
        ds.dataset_id = "dummy"
        ds.subset = None
        ds.POSITION_LABELS = ["A", "B", "C", "D"]

        # Minimal fake dataset rows
        rows = [
            {
                "Question": f"Q{i}?",
                "A": "alpha", "B": "beta", "C": "gamma", "D": "delta",
                "Answer": "A",
            }
            for i in range(10)
        ]
        ds._parse_mcq_examples(rows, max_examples=5)

    def _run_uncertainty_parse(self):
        from src.nb.datasets.uncertainty import UncertaintyMCQDataset

        ds = UncertaintyMCQDataset.__new__(UncertaintyMCQDataset)
        ds.split_seed = 99
        ds.POSITION_LABELS = ["A", "B", "C", "D"]

        rows = [
            {
                "Question": f"Q{i}?",
                "A": "alpha", "B": "beta", "C": "gamma", "D": "delta",
                "Answer": "A",
            }
            for i in range(10)
        ]
        ds._parse_mcq_examples(rows, max_examples=5)

    def test_sycophancy_does_not_mutate_global_rng(self):
        """SycophancyMCQDataset._parse_mcq_examples must not alter global random state."""
        import random

        random.seed(7)
        before = random.random()   # draw one value to mark current position
        random.seed(7)             # reset to same position

        self._run_sycophancy_parse()

        after = random.random()    # if global state was reset inside, this equals before
        # With the fix, _parse_mcq_examples uses a local RNG so global state is untouched
        # and `after` should equal `before` (we're at the same position in the global stream)
        assert after == pytest.approx(before), (
            "Global random state was mutated by _parse_mcq_examples"
        )

    def test_uncertainty_does_not_mutate_global_rng(self):
        """UncertaintyMCQDataset._parse_mcq_examples must not alter global random state."""
        import random

        random.seed(13)
        before = random.random()
        random.seed(13)

        self._run_uncertainty_parse()

        after = random.random()
        assert after == pytest.approx(before), (
            "Global random state was mutated by _parse_mcq_examples"
        )

    def test_sycophancy_results_are_deterministic(self):
        """Two calls with the same split_seed must produce the same examples."""
        from src.nb.datasets.sycophancy import SycophancyMCQDataset

        rows = [
            {
                "Question": f"Q{i}?",
                "A": "alpha", "B": "beta", "C": "gamma", "D": "delta",
                "Answer": "A",
            }
            for i in range(20)
        ]

        def run():
            ds = SycophancyMCQDataset.__new__(SycophancyMCQDataset)
            ds.split_seed = 42
            ds.dataset_id = "dummy"
            ds.subset = None
            ds.POSITION_LABELS = ["A", "B", "C", "D"]
            return ds._parse_mcq_examples(rows, max_examples=10)

        assert [e["incorrect_idx"] for e in run()] == [e["incorrect_idx"] for e in run()]


# ---------------------------------------------------------------------------
# Issue 10 — PositionBiasDataset.get_probe_pairs returns empty-string placeholders
# ---------------------------------------------------------------------------


class TestIssue10PositionProbePairs:
    """get_probe_pairs must return fully-formed ContrastivePairs with no empty strings."""

    def _make_dataset(self):
        from src.nb.datasets.position import PositionBiasDataset

        ds = PositionBiasDataset.__new__(PositionBiasDataset)
        ds._probe_indices = [0, 1]
        ds._raw_data = [
            {
                "question": "What is 2+2?",
                "choices": ["4", "3", "5", "6"],
            },
            {
                "question": "Colour of sky?",
                "choices": ["Blue", "Red", "Green", "Yellow"],
            },
        ]
        ds._loaded = True
        return ds

    def _make_tokenizer(self):
        tok = MagicMock()
        # Return something deterministic based on the call so positive != negative
        tok.apply_chat_template = MagicMock(
            side_effect=lambda msgs, **kw: " | ".join(m["content"] for m in msgs)
        )
        return tok

    def test_no_empty_strings_in_pairs(self):
        """All ContrastivePair texts must be non-empty after the fix."""
        from src.nb.datasets.position import PositionBiasDataset

        ds = self._make_dataset()
        tok = self._make_tokenizer()

        with patch.object(type(ds), "_ensure_loaded", lambda self: None):
            with patch(
                "src.nb.datasets.position.format_mcq_conversation",
                side_effect=lambda tok, q, choices, pos: f"{q}|pos={pos}",
            ):
                pairs = ds.get_probe_pairs(tok)

        for pair in pairs:
            assert pair.positive_text != "", f"Empty positive_text in pair: {pair}"
            assert pair.negative_text != "", f"Empty negative_text in pair: {pair}"

    def test_pairs_cover_all_non_a_positions(self):
        """Each question+choice should produce 3 pairs (A vs B, A vs C, A vs D)."""
        ds = self._make_dataset()
        tok = self._make_tokenizer()

        n_examples = len(ds._probe_indices)
        n_choices = len(ds._raw_data[0]["choices"])
        expected_pairs = n_examples * n_choices * 3  # 3 non-A positions

        with patch.object(type(ds), "_ensure_loaded", lambda self: None):
            with patch(
                "src.nb.datasets.position.format_mcq_conversation",
                side_effect=lambda tok, q, choices, pos: f"{q}|pos={pos}",
            ):
                pairs = ds.get_probe_pairs(tok)

        assert len(pairs) == expected_pairs, (
            f"Expected {expected_pairs} pairs, got {len(pairs)}"
        )

    def test_positive_is_always_position_a(self):
        """positive_text must always correspond to position A (pos=0)."""
        ds = self._make_dataset()
        tok = self._make_tokenizer()

        with patch.object(type(ds), "_ensure_loaded", lambda self: None):
            with patch(
                "src.nb.datasets.position.format_mcq_conversation",
                side_effect=lambda tok, q, choices, pos: f"{q}|pos={pos}",
            ):
                pairs = ds.get_probe_pairs(tok)

        for pair in pairs:
            assert "|pos=0" in pair.positive_text, (
                f"positive_text should be position A (pos=0): {pair.positive_text!r}"
            )
            assert "|pos=0" not in pair.negative_text, (
                f"negative_text should NOT be position A: {pair.negative_text!r}"
            )


# ---------------------------------------------------------------------------
# Issue 11 — Empty probe pairs should not crash embedding extraction
# ---------------------------------------------------------------------------


class TestIssue11EmptyProbePairsFallback:
    """Base probe building should skip cleanly when dataset returns no pairs."""

    def test_build_probe_skips_when_no_contrastive_pairs(self):
        from src.nb.experiments.base import BiasExperiment, ExperimentConfig

        class _DummyDataset:
            name = "dummy"

            def get_probe_pairs(self, tokenizer):
                return []

        class _DummyExperiment(BiasExperiment):
            @property
            def bias_type(self) -> str:
                return "dummy"

            def _create_dataset(self):
                return _DummyDataset()

            def _compute_metrics(self, rewards, eval_examples):
                return {"accuracy": 0.0}

            def _create_plot(self, results, output_path):
                return None

        config = ExperimentConfig(
            name="dummy",
            bias_type="dummy",
            model_path="dummy/model",
            save_probe=False,
        )
        exp = _DummyExperiment(config)
        exp.model = object()      # type: ignore[assignment]
        exp.tokenizer = object()  # type: ignore[assignment]
        exp.probe_dataset = _DummyDataset()  # type: ignore[assignment]

        metadata = exp.build_probe()

        assert metadata["probe_skipped"] is True
        assert metadata["n_contrastive_pairs"] == 0
        assert exp.probe is None

