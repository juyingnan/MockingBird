"""Temporal pooling for encoder hidden states."""

from __future__ import annotations

from typing import Any


def attention_masked_mean(hidden_state: Any, frame_attention_mask: Any) -> Any:
    """Mean-pool valid encoder frames while accumulating in float32."""
    import torch

    if hidden_state.ndim != 3:
        raise ValueError(f"hidden_state must have shape [batch, time, hidden], got {hidden_state.shape}")
    if frame_attention_mask.ndim != 2:
        raise ValueError(
            f"frame_attention_mask must have shape [batch, time], got {frame_attention_mask.shape}"
        )
    if hidden_state.shape[:2] != frame_attention_mask.shape:
        raise ValueError("hidden states and frame attention mask have incompatible shapes")

    mask = frame_attention_mask.to(device=hidden_state.device, dtype=torch.float32).unsqueeze(-1)
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("cannot pool an utterance with no valid encoder frames")
    return (hidden_state.float() * mask).sum(dim=1) / counts
