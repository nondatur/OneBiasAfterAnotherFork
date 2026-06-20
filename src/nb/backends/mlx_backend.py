"""
MLX (Apple Silicon) reward-model backend.

Runs the transformer **backbone** with `mlx` / `mlx-lm`, then applies the scalar
reward ``score`` head and all null-space projection in shared **CPU float32
torch** code (reusing :func:`src.nb.nullbias.probe.project_to_null_space`). This
confines numerical differences vs. the transformers reference to the backbone
forward pass alone.

Why a custom loader is needed:
- `mlx-lm` loads causal-LM checkpoints (a ``lm_head``); reward models are
  ``*ForSequenceClassification`` with a scalar ``score`` head and **no** usable
  ``lm_head``. So we build only the backbone from `mlx-lm`'s model registry and
  load the ``score`` weights separately into a torch ``nn.Linear``.
- The bias method only ever needs ``hidden_states[-1]`` (post-final-norm,
  last-layer). `mlx-lm` exposes exactly that as ``model.model(input_ids)`` — no
  deep instrumentation required.

Batching uses **right padding** so the causal mask alone makes each example's last
non-pad token independent of trailing pad tokens (per-example scores match the
transformers reference, which masks pads explicitly).
"""

from __future__ import annotations

import glob
import importlib
import json
import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.nb.backends.base import ModelBackend
from src.nb.nullbias.probe import project_to_null_space, tokenize_inputs

logger = logging.getLogger(__name__)

# mlx-lm modules that expose a reward-model backbone we support. Keyed by HF
# config ``model_type``.
_SUPPORTED_MODEL_TYPES = ("llama", "qwen2", "qwen3")


class MLXBackend(ModelBackend):
    """Reward-model inference on Apple Silicon via MLX."""

    def __init__(
        self,
        *,
        model_path: str,
        trust_remote_code: bool = True,
        mlx_quant: Optional[str] = None,
        max_length: int = 2048,
    ) -> None:
        import mlx.core as mx
        import mlx.nn as mlx_nn

        self._mx = mx
        self.model_path = model_path
        self.mlx_quant = mlx_quant
        self.max_length = max_length

        local_dir = self._resolve_model_dir(model_path)
        self.local_dir = local_dir

        # Tokenizer: HF tokenizer, reused for chat-template / pair formatting.
        # Force right padding so batched causal attention is correct (see module
        # docstring).
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        with open(os.path.join(local_dir, "config.json")) as f:
            cfg = json.load(f)
        self.model_type = cfg.get("model_type", "")
        self.hidden_size = int(cfg.get("hidden_size"))

        self.backbone, self.score_head = self._build(cfg, local_dir, mx, mlx_nn)
        self.hidden_dtype = "bfloat16" if mlx_quant is None else f"{mlx_quant}"
        logger.info(
            "MLX backend ready: %s (model_type=%s, precision=%s, hidden=%d)",
            model_path, self.model_type, self.hidden_dtype, self.hidden_size,
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _resolve_model_dir(model_path: str) -> str:
        if os.path.isdir(model_path):
            return model_path
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model_path,
            allow_patterns=[
                "*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.jinja",
            ],
        )

    def _build(self, cfg: dict, local_dir: str, mx, mlx_nn) -> Tuple[Any, torch.nn.Linear]:
        model_type = cfg.get("model_type", "")
        try:
            arch = importlib.import_module(f"mlx_lm.models.{model_type}")
        except ModuleNotFoundError as e:
            raise NotImplementedError(
                f"MLX backend does not support model_type={model_type!r}. "
                f"Supported reward-model backbones: {_SUPPORTED_MODEL_TYPES}."
            ) from e

        args = arch.ModelArgs.from_dict(cfg)
        full_model = arch.Model(args)          # CausalLM wrapper; we only use .model
        backbone = full_model.model            # inner backbone → post-norm hidden states

        # Merge all safetensors shards.
        weights: dict = {}
        shards = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No .safetensors weights found in {local_dir}")
        for shard in shards:
            weights.update(mx.load(shard))

        score_head = self._extract_score_head(weights, mx)

        # Backbone params live under the "model." prefix in the checkpoint.
        bb_weights = {
            k[len("model."):]: v for k, v in weights.items() if k.startswith("model.")
        }
        # Let the arch's sanitize() drop/remap keys (e.g. rotary inv_freq) if present.
        if hasattr(full_model, "sanitize"):
            prefixed = {f"model.{k}": v for k, v in bb_weights.items()}
            try:
                prefixed = full_model.sanitize(prefixed)
                bb_weights = {
                    k[len("model."):]: v for k, v in prefixed.items() if k.startswith("model.")
                }
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("mlx-lm sanitize() failed (%s); loading raw keys.", exc)

        backbone.load_weights(list(bb_weights.items()), strict=False)

        if self.mlx_quant in ("4bit", "8bit"):
            bits = 4 if self.mlx_quant == "4bit" else 8
            mlx_nn.quantize(backbone, group_size=64, bits=bits)
        else:
            backbone.set_dtype(mx.bfloat16)

        backbone.eval()
        mx.eval(backbone.parameters())
        return backbone, score_head

    def _extract_score_head(self, weights: dict, mx) -> torch.nn.Linear:
        """Load the scalar reward head ([1, hidden]) into a CPU fp32 torch Linear."""
        w_key = None
        for cand in ("score.weight", "v_head.summary.weight", "v_head.weight", "classifier.weight"):
            if cand in weights:
                w_key = cand
                break
        if w_key is None:
            for k, v in weights.items():
                if "score" in k and getattr(v, "ndim", 0) == 2 and v.shape[0] == 1:
                    w_key = k
                    break
        if w_key is None:
            raise ValueError(
                "MLX backend: could not locate the reward 'score' head in the checkpoint. "
                f"Top-level keys sampled: {sorted(k for k in weights if 'score' in k or 'head' in k)[:8]}"
            )

        w = torch.from_numpy(np.array(weights[w_key].astype(mx.float32))).float()  # [1, hidden]
        out_features, in_features = w.shape
        b_key = w_key.rsplit(".", 1)[0] + ".bias"
        has_bias = b_key in weights
        linear = torch.nn.Linear(in_features, out_features, bias=has_bias)
        with torch.no_grad():
            linear.weight.copy_(w)
            if has_bias:
                b = torch.from_numpy(np.array(weights[b_key].astype(mx.float32))).float()
                linear.bias.copy_(b)
        linear.eval()
        logger.info("Loaded reward score head from '%s' (out=%d, in=%d, bias=%s).",
                    w_key, out_features, in_features, has_bias)
        return linear

    # --------------------------------------------------------------- forward
    def _last_hidden_batch(self, batch_texts: List[Any], max_length: int) -> torch.Tensor:
        """Return [B, hidden] last-token post-norm hidden states (CPU float32)."""
        mx = self._mx
        inputs = tokenize_inputs(
            self.tokenizer,
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]              # torch [B, T] (right-padded)
        attention_mask = inputs["attention_mask"]    # torch [B, T]

        ids_mx = mx.array(input_ids.numpy().astype(np.int32))
        hidden = self.backbone(ids_mx)               # mx [B, T, d] post-norm
        mx.eval(hidden)
        hidden_t = torch.from_numpy(np.array(hidden.astype(mx.float32)))  # [B, T, d] cpu fp32

        last_idx = attention_mask.sum(dim=1) - 1     # [B]
        rows = torch.arange(hidden_t.shape[0])
        return hidden_t[rows, last_idx]              # [B, d] cpu fp32

    def _iter_batches(self, texts, batch_size, max_length, show_progress, desc):
        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc=desc, total=n_batches)
        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            yield self._last_hidden_batch(texts[start:end], max_length)

    # --------------------------------------------------------- public API
    def embed_texts(
        self, texts, *, batch_size=8, max_length=2048, show_progress=True
    ) -> torch.Tensor:
        chunks = list(
            self._iter_batches(texts, batch_size, max_length, show_progress, "Extracting embeddings (MLX)")
        )
        return torch.cat(chunks, dim=0)

    def _apply_score(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.score_head(hidden).squeeze(-1)

    def score_texts_both(
        self, texts, *, probe=None, null_alpha=1.0, batch_size=8, max_length=2048, show_progress=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        baseline_chunks: List[torch.Tensor] = []
        nulled_chunks: List[torch.Tensor] = []
        do_nulling = probe is not None
        for last_hidden in self._iter_batches(
            texts, batch_size, max_length, show_progress, "Computing rewards (MLX, baseline+nulled)"
        ):
            baseline_chunks.append(self._apply_score(last_hidden))
            if do_nulling:
                nulled_hidden = project_to_null_space(last_hidden, probe, alpha=null_alpha)
                nulled_chunks.append(self._apply_score(nulled_hidden))
            else:
                nulled_chunks.append(baseline_chunks[-1])
        return torch.cat(baseline_chunks, dim=0), torch.cat(nulled_chunks, dim=0)

    def score_texts(
        self, texts, *, probe=None, null_alpha=1.0, batch_size=8, max_length=2048, show_progress=True
    ) -> torch.Tensor:
        scores: List[torch.Tensor] = []
        do_nulling = probe is not None
        for last_hidden in self._iter_batches(
            texts, batch_size, max_length, show_progress, "Computing rewards (MLX)"
        ):
            hidden = project_to_null_space(last_hidden, probe, alpha=null_alpha) if do_nulling else last_hidden
            scores.append(self._apply_score(hidden))
        return torch.cat(scores, dim=0)
