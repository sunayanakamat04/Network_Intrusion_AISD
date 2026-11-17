"""Configuration and feature schema constants for the G2 Sequence / Attention experiment.

Single source of truth: imports the exact 66 feature columns and 6 envelope metadata columns
from Pipeline v2 (`features_v2.py`) without duplicating feature definitions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import sys
from typing import Any

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

import features_v2 as _fv  # noqa: E402

FEATURE_COLUMNS: tuple[str, ...] = tuple(_fv.FEATURES)
ENVELOPE_COLUMNS: tuple[str, ...] = tuple(_fv.ENVELOPE)
GROUP_COLUMN: str = "src_ip"
ORDER_COLUMNS: tuple[str, str] = ("bidirectional_last_seen_ms", "bidirectional_first_seen_ms")
ROUND_COLUMN: str = "round"
LABEL_COLUMN: str = "label"
NORMAL_CLASS: str = "normal"

DEFAULT_DATA_PATH: str = os.path.join(V2_DIR, "data", "flows", "all_flows.csv")
DEFAULT_ARTIFACTS_DIR: str = os.path.join(EXPERIMENT_DIR, "artifacts")
DEFAULT_SHIPPED_XGB_MODEL: str = os.path.join(V2_DIR, "models", "best_xgb.json")
DEFAULT_SHIPPED_XGB_META: str = os.path.join(V2_DIR, "models", "best_meta.json")


@dataclass(frozen=True)
class G2Config:
    """Hyperparameters and paths for the G2 Causal LSTM-Attention experiment."""

    data_path: str = DEFAULT_DATA_PATH
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
    train_rounds: tuple[int, ...] = (3, 4, 5)
    test_round: int = 6
    classes: tuple[str, ...] = ("bruteforce", "normal", "scan", "synflood")
    seq_len: int = 8
    input_dim: int = len(FEATURE_COLUMNS)
    embed_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    proj_dropout: float = 0.2
    lstm_dropout: float = 0.3
    head_dropout: float = 0.3
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 8
    grad_clip: float = 1.0
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["train_rounds"] = list(self.train_rounds)
        d["classes"] = list(self.classes)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "G2Config":
        payload = dict(data)
        if "train_rounds" in payload:
            payload["train_rounds"] = tuple(int(r) for r in payload["train_rounds"])
        if "classes" in payload:
            payload["classes"] = tuple(str(c) for c in payload["classes"])
        return cls(**payload)
