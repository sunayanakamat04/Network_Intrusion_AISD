"""Leak-free causal sequence construction and stateful single-flow causal buffer.

Design guarantees:
- Independent traffic streams are grouped strictly by (round, src_ip).
- `src_ip`, ports, and timestamps are NEVER placed in the 66-feature tensor.
- Within each stream, flows are sorted chronologically by
  (bidirectional_last_seen_ms, bidirectional_first_seen_ms) — the exact causal arrival
  order in which NFStream emits completed flows during live detection.
- Valid timesteps are placed contiguously at indices [0 .. L_eff - 1] and padded with zeros
  at [L_eff .. seq_len - 1] so PyTorch's `pack_padded_sequence` skips every padded step
  with zero impact on LSTM hidden states.
- Every flow in the input table produces exactly one causal sequence (100% row preservation
  for exact 1-to-1 evaluation against XGBoost on Round 6's 2,735 flows).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any
import numpy as np
import pandas as pd

import config as g2_cfg


def build_causal_sequences(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    y_encoded: np.ndarray | None = None,
    seq_len: int = 8,
) -> dict[str, Any]:
    """Construct causal many-to-one flow sequences strictly inside each (round, src_ip) stream."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if len(df) != len(X_scaled):
        raise ValueError(f"Row count mismatch: len(df)={len(df)} vs len(X_scaled)={len(X_scaled)}")
    if X_scaled.ndim != 2 or X_scaled.shape[1] != len(g2_cfg.FEATURE_COLUMNS):
        raise ValueError(
            f"Expected X_scaled shape (N, {len(g2_cfg.FEATURE_COLUMNS)}), got {X_scaled.shape}"
        )
    if y_encoded is not None and len(y_encoded) != len(df):
        raise ValueError(f"Label length mismatch: len(y_encoded)={len(y_encoded)} vs len(df)={len(df)}")

    n_rows = len(df)
    n_feats = X_scaled.shape[1]

    X_seq = np.zeros((n_rows, seq_len, n_feats), dtype=np.float32)
    lengths = np.zeros(n_rows, dtype=np.int64)
    mask = np.zeros((n_rows, seq_len), dtype=bool)
    y_out = np.zeros(n_rows, dtype=np.int64) if y_encoded is not None else None
    orig_indices = np.zeros(n_rows, dtype=np.int64)
    rounds_out = np.zeros(n_rows, dtype=np.int64)
    last_seen_out = np.zeros(n_rows, dtype=np.int64)

    # Work on positional indices 0..N-1 to remain independent of DataFrame index labels
    work = pd.DataFrame(
        {
            "_pos": np.arange(n_rows, dtype=np.int64),
            g2_cfg.ROUND_COLUMN: df[g2_cfg.ROUND_COLUMN].to_numpy(dtype=np.int64),
            g2_cfg.GROUP_COLUMN: df[g2_cfg.GROUP_COLUMN].astype(str).to_numpy(),
            "last_ms": df[g2_cfg.ORDER_COLUMNS[0]].to_numpy(dtype=np.int64),
            "first_ms": df[g2_cfg.ORDER_COLUMNS[1]].to_numpy(dtype=np.int64),
        }
    )

    out_idx = 0
    # Group strictly by (round, src_ip) so no sequence ever crosses a round or source-IP boundary
    for (rnd, _src_ip), grp in work.groupby([g2_cfg.ROUND_COLUMN, g2_cfg.GROUP_COLUMN], sort=True):
        grp_sorted = grp.sort_values(
            by=["last_ms", "first_ms", "_pos"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        pos_arr = grp_sorted["_pos"].to_numpy(dtype=np.int64)
        last_arr = grp_sorted["last_ms"].to_numpy(dtype=np.int64)
        stream_feats = X_scaled[pos_arr]
        stream_len = len(pos_arr)

        for i in range(stream_len):
            start = max(0, i - seq_len + 1)
            end = i + 1
            eff_len = end - start
            # Place valid steps contiguously at [0 : eff_len]; [eff_len : seq_len] remains 0.0
            X_seq[out_idx, :eff_len, :] = stream_feats[start:end]
            lengths[out_idx] = eff_len
            mask[out_idx, :eff_len] = True
            orig_indices[out_idx] = pos_arr[i]
            rounds_out[out_idx] = int(rnd)
            last_seen_out[out_idx] = int(last_arr[i])
            if y_out is not None and y_encoded is not None:
                y_out[out_idx] = int(y_encoded[pos_arr[i]])
            out_idx += 1

    assert out_idx == n_rows
    return {
        "X": X_seq,
        "lengths": lengths,
        "mask": mask,
        "y": y_out,
        "orig_indices": orig_indices,
        "rounds": rounds_out,
        "last_seen_ms": last_seen_out,
    }


class CausalFlowBuffer:
    """Stateful per-source-IP causal window buffer for single-flow live inference and latency tests.

    `src_ip` is used strictly as a hash key to route each host's independent flow stream;
    it never enters the 66-feature numerical array passed to the model.
    """

    def __init__(self, seq_len: int = 8, feature_dim: int = len(g2_cfg.FEATURE_COLUMNS)) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self._streams: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=seq_len))
        self._buf = np.zeros((1, seq_len, feature_dim), dtype=np.float32)

    def reset(self) -> None:
        self._streams.clear()
        self._buf.fill(0.0)

    def push_and_build(self, src_ip: str, scaled_features: np.ndarray) -> tuple[np.ndarray, int]:
        """Append one preprocessed 66-feature flow for `src_ip` and return (X_seq_1, eff_len)."""
        vec = np.asarray(scaled_features, dtype=np.float32).reshape(self.feature_dim)
        stream = self._streams[str(src_ip)]
        stream.append(vec)
        eff_len = len(stream)

        out = np.zeros((1, self.seq_len, self.feature_dim), dtype=np.float32)
        for idx, item in enumerate(stream):
            out[0, idx, :] = item
        return out, eff_len
