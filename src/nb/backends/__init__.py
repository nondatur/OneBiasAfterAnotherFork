"""
Execution backends for reward-model inference.

Public API:
- :class:`ModelBackend` — the backend interface (see :mod:`.base`).
- :func:`select_backend` — resolve a ``device`` string to a backend kind.
- :func:`create_backend` — build the model/backend + tokenizer for a config.

Design: the CUDA/CPU ``transformers`` path is returned as the *raw* HF model
(unchanged behaviour); only the Apple-Silicon MLX path is a :class:`ModelBackend`
instance. The free functions in :mod:`src.nb.nullbias.probe` dispatch to a backend
when their ``model`` argument is one, so every experiment routes through MLX with
no per-call-site changes. Heavy imports (``transformers``, ``mlx``) are done
lazily inside :func:`create_backend` so importing this package never pulls in a
framework that isn't installed.
"""

from __future__ import annotations

import logging
import platform
from typing import Any, Tuple

from src.nb.backends.base import ModelBackend

logger = logging.getLogger(__name__)

__all__ = ["ModelBackend", "select_backend", "create_backend"]


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def select_backend(device: str) -> str:
    """Resolve a ``device`` string to a backend kind: ``"mlx"`` or ``"transformers"``.

    - ``"mlx"`` → MLX (forced; errors later if unavailable).
    - ``"auto"`` → **unchanged on CUDA hosts** (CUDA present → transformers); else
      MLX on Apple Silicon when available; else transformers (CPU).
    - ``"cuda"`` / ``"cuda:N"`` / ``"cpu"`` / anything else → transformers.

    Pure function with no model side effects.
    """
    d = (device or "").strip().lower()
    if d == "mlx":
        return "mlx"
    if d == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "transformers"
        except Exception:  # pragma: no cover - torch always present, defensive
            pass
        if _is_apple_silicon() and _mlx_available():
            logger.info("device=auto resolved to MLX backend (Apple Silicon, no CUDA).")
            return "mlx"
        return "transformers"
    return "transformers"


def create_backend(config: Any) -> Tuple[Any, Any]:
    """Build the model (or :class:`ModelBackend`) and tokenizer for ``config``.

    Returns ``(model_or_backend, tokenizer)``:
    - transformers: ``(AutoModelForSequenceClassification, AutoTokenizer)`` — the
      exact objects/behaviour the pipeline used before backends existed.
    - mlx: ``(MLXBackend, tokenizer)``.

    ``config`` must expose ``model_path``, ``trust_remote_code``, ``device`` and
    (optionally) ``mlx_quant`` and ``max_length``.
    """
    kind = select_backend(config.device)
    if kind == "mlx":
        from src.nb.backends.mlx_backend import MLXBackend

        backend = MLXBackend(
            model_path=config.model_path,
            trust_remote_code=getattr(config, "trust_remote_code", True),
            mlx_quant=getattr(config, "mlx_quant", None),
            max_length=getattr(config, "max_length", 2048),
        )
        return backend, backend.tokenizer

    return _load_transformers(config)


def _load_transformers(config: Any) -> Tuple[Any, Any]:
    """Load an HF sequence-classification reward model + tokenizer.

    This reproduces the original ``BiasExperiment.load_model`` body exactly so the
    CUDA/CPU paths are byte-for-byte unchanged.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pin RIGHT padding. Every activation read in probe.py gathers the last real token via
    # `attention_mask.sum(dim=1) - 1`, which is only the last token under right padding; with
    # left padding that index lands mid-sequence on a pad and the scores are silently wrong.
    # The MLX backend (mlx_backend.py:79) and parity_check.py:83 already pin it; this path did
    # not, so it inherited each checkpoint's tokenizer default -- fine for Qwen3 ('right'), but
    # several Llama tokenizers default to 'left'. It also keeps the baseline branch, which uses
    # HF's own pad_token_id-based pooling, consistent with the nulled branch's manual gather.
    tokenizer.padding_side = "right"

    # "cuda" / "auto" → let HF shard the model across all visible GPUs.
    # An explicit single-device string (e.g. "cuda:0", "cpu") is passed
    # through unchanged so the caller retains control.
    device_map = "auto" if config.device in ("cuda", "auto") else config.device
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    # .to() intentionally omitted: device_map handles placement.

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
