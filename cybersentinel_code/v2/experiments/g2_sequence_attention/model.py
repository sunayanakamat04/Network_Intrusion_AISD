"""G2 Causal LSTM with Masked Temporal Attention architecture (PyTorch).

Research grounding:
- Unidirectional stacked LSTM temporal encoder (Berhanu et al., 2025), extended to multiclass
  and strict causal sequence-length packing via `pack_padded_sequence` so padded timesteps
  have mathematically ZERO influence on LSTM hidden states.
- Current-step-anchored Masked Temporal Attention readout + multi-layer ClassificationHead
  inspired by attention-based NIDS architectures (Zhu et al., 2025; Zaim & Yolacan, 2025).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CausalLSTMAttention(nn.Module):
    """Causal LSTM + Current-Step-Anchored Masked Temporal Attention for multiclass NIDS."""

    def __init__(
        self,
        input_dim: int = 66,
        num_classes: int = 4,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        proj_dropout: float = 0.2,
        lstm_dropout: float = 0.3,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Scaled dot-product temporal attention anchored on the current (final valid) flow h_t
        self.attn_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._attn_scale = math.sqrt(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run causal sequence classification with exact padded-timestep exclusion.

        Args:
            x: Input tensor of shape (B, L, input_dim), with valid flows at [0 .. length - 1].
            lengths: Valid sequence lengths of shape (B,), values in [1, L].
            return_attention: If True, returns (logits, attention_weights).
        """
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError(f"Expected x shape (B, L, {self.input_dim}), got {tuple(x.shape)}")
        batch_size, max_len, _ = x.shape
        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError(f"Expected lengths shape ({batch_size},), got {tuple(lengths.shape)}")
        if int(lengths.min().item()) < 1 or int(lengths.max().item()) > max_len:
            raise ValueError(f"Sequence lengths must lie in [1, {max_len}]")

        device = x.device
        lengths_dev = lengths.to(device=device, dtype=torch.long)
        steps = torch.arange(max_len, device=device).unsqueeze(0)  # (1, L)
        valid_mask = steps < lengths_dev.unsqueeze(1)              # (B, L)

        # Position-wise projection; explicitly zero padded slots before packing
        proj = self.input_proj(x)
        proj = proj * valid_mask.unsqueeze(-1).to(proj.dtype)

        # Pack padded sequences so LSTM NEVER processes padded timesteps (zero hidden-state drift)
        packed = pack_padded_sequence(
            proj,
            lengths_dev.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=max_len,
        )  # (B, L, hidden_dim), padded positions are strictly 0.0

        # Extract the exact hidden state of the current (last valid) flow h_t
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = lengths_dev - 1
        h_current = lstm_out[batch_idx, last_idx, :]  # (B, hidden_dim)

        # Current-step-anchored Masked Temporal Attention over valid causal timesteps only
        q = self.attn_query(h_current).unsqueeze(1)            # (B, 1, hidden_dim)
        k = self.attn_key(lstm_out)                            # (B, L, hidden_dim)
        scores = (q * k).sum(dim=-1) / self._attn_scale        # (B, L)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)           # (B, L), strictly 0.0 on padding
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, hidden_dim)

        fused = torch.cat([h_current, context], dim=-1)        # (B, 2 * hidden_dim)
        logits = self.classifier(fused)                        # (B, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
