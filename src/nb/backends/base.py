"""
Execution-backend interface for reward-model inference.

The bias-evaluation pipeline only needs a model to turn batches of text into two
quantities:

1. **Last-token, post-final-norm hidden states** (the residual-stream activation
   that feeds the reward ``score`` head — equivalent to HF
   ``output_hidden_states=True`` then ``hidden_states[-1]`` at the last non-pad
   token).
2. **Scalar reward scores**, optionally with the probe subspace projected out of
   the hidden state before the score head is applied.

A :class:`ModelBackend` provides exactly these primitives. **All** null-space
linear algebra (``project_to_null_space``, ``gram_schmidt``, difference-of-means
probes, metrics) stays in :mod:`src.nb.nullbias.probe` and runs on CPU ``float32``
torch tensors, so it is identical regardless of which backend produced the hidden
states. This keeps numerical differences between backends confined to the
transformer forward pass alone.

The default CUDA/CPU path uses the HuggingFace ``transformers`` model directly
(see :func:`src.nb.backends.create_backend`) — those code paths are unchanged and
do **not** go through a backend object. Only the additive MLX (Apple Silicon)
path is implemented as a :class:`ModelBackend`; the free functions in
``probe.py`` dispatch to it when their ``model`` argument is a backend instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import torch


class ModelBackend(ABC):
    """A pluggable reward-model execution backend.

    Implementations must return **CPU ``float32`` torch tensors** so the
    downstream probe / null-space math is byte-for-byte backend-agnostic.
    """

    #: HuggingFace tokenizer used for chat-template / pair formatting. Shared
    #: across backends (mlx-lm reuses HF tokenizers).
    tokenizer: Any

    @abstractmethod
    def embed_texts(
        self,
        texts: List[Any],
        *,
        batch_size: int = 8,
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Return ``[N, hidden_dim]`` last-token post-norm hidden states (CPU float32)."""

    @abstractmethod
    def score_texts(
        self,
        texts: List[Any],
        *,
        probe: Optional[torch.Tensor] = None,
        null_alpha: float = 1.0,
        batch_size: int = 8,
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Return ``[N]`` reward scores.

        Nulled scores when ``probe`` is given (project the probe subspace out of
        the hidden state before the score head), otherwise baseline scores.
        Mirrors :func:`src.nb.nullbias.probe.get_rewards_with_nulling`.
        """

    @abstractmethod
    def score_texts_both(
        self,
        texts: List[Any],
        *,
        probe: Optional[torch.Tensor] = None,
        null_alpha: float = 1.0,
        batch_size: int = 8,
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(baseline[N], nulled[N])`` from a single forward pass.

        Mirrors :func:`src.nb.nullbias.probe.get_rewards_both`. When ``probe`` is
        ``None`` the nulled tensor equals the baseline tensor.
        """
