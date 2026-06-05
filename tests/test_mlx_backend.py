"""
Tests for the execution-backend abstraction (src/nb/backends) and MLX dispatch.

Three layers:
- ``select_backend`` device routing — pure logic, always runs.
- probe.py free-function dispatch to a fake ModelBackend — no MLX needed.
- An opt-in end-to-end MLX smoke test (downloads a small reward model) gated by
  both mlx being installed and RUN_MLX_SMOKE=1.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

import pytest
import torch

from src.nb.backends import select_backend
from src.nb.backends.base import ModelBackend

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)


# --------------------------------------------------------------------------
# select_backend routing
# --------------------------------------------------------------------------
class TestSelectBackend:
    def test_mlx_is_forced(self):
        assert select_backend("mlx") == "mlx"

    @pytest.mark.parametrize("dev", ["cuda", "cuda:0", "cuda:1", "cpu", "", "weird"])
    def test_non_auto_non_mlx_is_transformers(self, dev):
        assert select_backend(dev) == "transformers"

    def test_auto_prefers_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert select_backend("auto") == "transformers"

    def test_auto_uses_mlx_on_apple_silicon_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("src.nb.backends._is_apple_silicon", return_value=True), \
             patch("src.nb.backends._mlx_available", return_value=True):
            assert select_backend("auto") == "mlx"

    def test_auto_falls_back_to_transformers_without_cuda_or_mlx(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("src.nb.backends._is_apple_silicon", return_value=False):
            assert select_backend("auto") == "transformers"


# --------------------------------------------------------------------------
# probe.py dispatch to a backend (no MLX required)
# --------------------------------------------------------------------------
class _FakeBackend(ModelBackend):
    """Records calls and returns deterministic tensors."""

    def __init__(self, hidden_dim: int = 8):
        self.tokenizer = object()
        self.hidden_dim = hidden_dim
        self.calls: List[str] = []

    def embed_texts(self, texts, *, batch_size=8, max_length=2048, show_progress=True):
        self.calls.append("embed_texts")
        return torch.arange(len(texts) * self.hidden_dim, dtype=torch.float32).reshape(
            len(texts), self.hidden_dim
        )

    def score_texts(self, texts, *, probe=None, null_alpha=1.0, batch_size=8, max_length=2048, show_progress=True):
        self.calls.append("score_texts")
        return torch.ones(len(texts))

    def score_texts_both(self, texts, *, probe=None, null_alpha=1.0, batch_size=8, max_length=2048, show_progress=True):
        self.calls.append("score_texts_both")
        n = len(texts)
        return torch.ones(n), torch.zeros(n)


class TestProbeDispatch:
    def test_get_embeddings_dispatches(self):
        from src.nb.nullbias.probe import get_embeddings

        be = _FakeBackend()
        out = get_embeddings(be, be.tokenizer, ["a", "b", "c"], show_progress=False)
        assert be.calls == ["embed_texts"]
        assert out.shape == (3, 8)

    def test_get_rewards_with_nulling_dispatches(self):
        from src.nb.nullbias.probe import get_rewards_with_nulling

        be = _FakeBackend()
        out = get_rewards_with_nulling(be, be.tokenizer, ["a", "b"], probe=torch.randn(8), show_progress=False)
        assert be.calls == ["score_texts"]
        assert out.shape == (2,)

    def test_get_rewards_both_dispatches(self):
        from src.nb.nullbias.probe import get_rewards_both

        be = _FakeBackend()
        base, nulled = get_rewards_both(be, be.tokenizer, ["a", "b"], probe=torch.randn(8), show_progress=False)
        assert be.calls == ["score_texts_both"]
        assert base.shape == (2,) and nulled.shape == (2,)

    def test_build_probe_direction_routes_through_backend(self):
        """build_probe_direction is unchanged but must transparently use the backend."""
        from src.nb.datasets.base import ContrastivePair
        from src.nb.nullbias.probe import build_probe_direction

        be = _FakeBackend()
        pairs = [ContrastivePair(positive_text="p", negative_text="n") for _ in range(3)]
        probe, meta = build_probe_direction(be, be.tokenizer, pairs)
        # two embed calls (positive + negative)
        assert be.calls == ["embed_texts", "embed_texts"]
        assert probe.shape == (8,)
        assert meta["hidden_dim"] == 8


# --------------------------------------------------------------------------
# opt-in end-to-end MLX smoke test
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_MLX, reason="mlx / mlx_lm not installed")
@pytest.mark.skipif(os.environ.get("RUN_MLX_SMOKE") != "1", reason="set RUN_MLX_SMOKE=1 to run (downloads a model)")
def test_mlx_backend_end_to_end():
    """MLXBackend produces finite rewards and post-norm hidden states."""
    from src.nb.backends.mlx_backend import MLXBackend

    backend = MLXBackend(model_path="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    texts = ["Question: 2+2?\nAnswer: 4", "Question: 2+2?\nAnswer: 5"]
    emb = backend.embed_texts(texts, batch_size=2, show_progress=False)
    assert emb.shape == (2, backend.hidden_size)
    assert torch.isfinite(emb).all()

    base, nulled = backend.score_texts_both(texts, probe=None, batch_size=2, show_progress=False)
    assert base.shape == (2,)
    assert torch.isfinite(base).all()
    assert torch.allclose(base, nulled)  # probe=None → nulled == baseline
