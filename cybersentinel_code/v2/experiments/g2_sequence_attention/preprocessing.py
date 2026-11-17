"""Dataset loading, round-based splitting, leak-free feature scaling, and class weighting."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import pandas as pd

import config as g2_cfg


def validate_feature_columns(columns: Sequence[str]) -> None:
    """Verify that model input columns are strictly the 66 Pipeline v2 behavioral features."""
    cols = list(columns)
    forbidden = set(g2_cfg.ENVELOPE_COLUMNS) | {g2_cfg.ROUND_COLUMN, g2_cfg.LABEL_COLUMN}
    leaked = forbidden.intersection(cols)
    if leaked:
        raise ValueError(f"Forbidden metadata/target columns found in model features: {sorted(leaked)}")
    if len(cols) != len(g2_cfg.FEATURE_COLUMNS) or cols != list(g2_cfg.FEATURE_COLUMNS):
        raise ValueError(
            f"Expected exact {len(g2_cfg.FEATURE_COLUMNS)} Pipeline v2 features in canonical order; "
            f"got {len(cols)} columns."
        )


def load_flows_csv(path: str | None = None) -> pd.DataFrame:
    """Load the shared 10,676-flow CSV and validate required columns without modifying it."""
    csv_path = path or g2_cfg.DEFAULT_DATA_PATH
    df = pd.read_csv(csv_path)
    required = (
        {g2_cfg.ROUND_COLUMN, g2_cfg.LABEL_COLUMN}
        | set(g2_cfg.ENVELOPE_COLUMNS)
        | set(g2_cfg.FEATURE_COLUMNS)
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def split_by_round(
    df: pd.DataFrame,
    train_rounds: Sequence[int] = (3, 4, 5),
    test_round: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform strict round-based splitting with zero train/test round overlap."""
    tr_set = {int(r) for r in train_rounds}
    te_val = int(test_round)
    if te_val in tr_set:
        raise ValueError(f"Data leakage: test_round={te_val} overlaps train_rounds={sorted(tr_set)}")

    train_df = df[df[g2_cfg.ROUND_COLUMN].isin(tr_set)].copy().reset_index(drop=True)
    test_df = df[df[g2_cfg.ROUND_COLUMN] == te_val].copy().reset_index(drop=True)
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Empty split produced (train={len(train_df)}, test={len(test_df)}) "
            f"for train_rounds={sorted(tr_set)}, test_round={te_val}"
        )
    return train_df, test_df


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """Smooth monotone compression for heavy-tailed network flow counters and rates."""
    return np.sign(x) * np.log1p(np.abs(x))


def _sanitize_numeric(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/+inf/-inf with 0.0 in float64 precision before transformation."""
    return np.nan_to_num(arr.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


class FlowFeaturePreprocessor:
    """Leak-free preprocessor fitted exclusively on training flows (66 features only).

    Applies non-finite sanitization -> signed log1p -> z-score standardization with
    zero-variance protection so constant TCP flag columns scale safely to 0.0.
    """

    def __init__(self, feature_columns: Sequence[str] = g2_cfg.FEATURE_COLUMNS) -> None:
        validate_feature_columns(feature_columns)
        self.feature_columns: tuple[str, ...] = tuple(feature_columns)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.constant_mask_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def _extract_matrix(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            # Strictly select ONLY the 66 validated behavioral feature columns.
            arr = data.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64, copy=True)
        else:
            arr = np.asarray(data, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Expected 2D array with {len(self.feature_columns)} columns, got shape {arr.shape}"
                )
        return _sanitize_numeric(arr)

    def fit(self, train_data: pd.DataFrame | np.ndarray) -> "FlowFeaturePreprocessor":
        """Fit scaling statistics strictly on training data."""
        raw = self._extract_matrix(train_data)
        transformed = _signed_log1p(raw)
        mean = transformed.mean(axis=0)
        std = transformed.std(axis=0)
        constant_mask = std < 1e-12
        safe_std = np.where(constant_mask, 1.0, std)

        self.mean_ = mean.astype(np.float64)
        self.std_ = safe_std.astype(np.float64)
        self.constant_mask_ = constant_mask
        self.is_fitted_ = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform a batch of flows into a finite float32 (N, 66) array."""
        if not self.is_fitted_ or self.mean_ is None or self.std_ is None or self.constant_mask_ is None:
            raise RuntimeError("FlowFeaturePreprocessor must be fitted on training data before transform().")
        raw = self._extract_matrix(data)
        transformed = _signed_log1p(raw)
        scaled = (transformed - self.mean_) / self.std_
        scaled[:, self.constant_mask_] = 0.0
        return scaled.astype(np.float32)

    def transform_single(self, flow: Mapping[str, float] | np.ndarray) -> np.ndarray:
        """Fast single-flow (66,) float32 transform without pandas overhead for live latency tests."""
        if not self.is_fitted_ or self.mean_ is None or self.std_ is None or self.constant_mask_ is None:
            raise RuntimeError("FlowFeaturePreprocessor must be fitted before transform_single().")
        if isinstance(flow, Mapping):
            raw = np.fromiter(
                (float(flow[k]) for k in self.feature_columns),
                dtype=np.float64,
                count=len(self.feature_columns),
            )
        else:
            raw = np.asarray(flow, dtype=np.float64).reshape(-1)
            if raw.shape[0] != len(self.feature_columns):
                raise ValueError(
                    f"Expected {len(self.feature_columns)} features, got {raw.shape[0]}"
                )
        raw = _sanitize_numeric(raw)
        transformed = _signed_log1p(raw)
        scaled = (transformed - self.mean_) / self.std_
        scaled[self.constant_mask_] = 0.0
        return scaled.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        if not self.is_fitted_ or self.mean_ is None or self.std_ is None or self.constant_mask_ is None:
            raise RuntimeError("Cannot serialize an unfitted FlowFeaturePreprocessor.")
        return {
            "feature_columns": list(self.feature_columns),
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "constant_mask": self.constant_mask_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FlowFeaturePreprocessor":
        obj = cls(feature_columns=state["feature_columns"])
        obj.mean_ = np.asarray(state["mean"], dtype=np.float64)
        obj.std_ = np.asarray(state["std"], dtype=np.float64)
        obj.constant_mask_ = np.asarray(state["constant_mask"], dtype=bool)
        obj.is_fitted_ = True
        return obj


def encode_labels(labels: Sequence[str] | np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Map string labels to deterministic integer class indices."""
    mapping = {c: i for i, c in enumerate(classes)}
    arr = np.asarray(labels)
    unknown = set(np.unique(arr).tolist()) - set(mapping.keys())
    if unknown:
        raise ValueError(f"Unknown labels encountered: {sorted(unknown)}")
    return np.array([mapping[str(lbl)] for lbl in arr], dtype=np.int64)


def decode_labels(indices: Sequence[int] | np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Map integer class indices back to string class names."""
    cls_arr = np.asarray(classes, dtype=object)
    idx_arr = np.asarray(indices, dtype=np.int64)
    return cls_arr[idx_arr]


def compute_class_weights(
    y_train_indices: np.ndarray,
    classes: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute balanced inverse-frequency class weights strictly from training labels.

    Formula: w_c = N / (K * N_c), matching sklearn's balanced weighting so each class
    contributes equally to the expected training loss without fabricating synthetic sequences.
    """
    y = np.asarray(y_train_indices, dtype=np.int64)
    num_classes = len(classes)
    total = int(len(y))
    counts = np.bincount(y, minlength=num_classes)
    if np.any(counts == 0):
        missing = [classes[i] for i, c in enumerate(counts) if c == 0]
        raise ValueError(f"Training split missing classes: {missing}")

    weights = (total / (num_classes * counts.astype(np.float64))).astype(np.float32)
    weighted_mass = counts.astype(np.float64) * weights.astype(np.float64)
    effective_props = weighted_mass / weighted_mass.sum()

    report = {
        "total_train_flows": total,
        "raw_counts": {classes[i]: int(counts[i]) for i in range(num_classes)},
        "raw_proportions": {classes[i]: round(float(counts[i] / total), 6) for i in range(num_classes)},
        "class_weights": {classes[i]: round(float(weights[i]), 6) for i in range(num_classes)},
        "effective_weighted_proportions": {
            classes[i]: round(float(effective_props[i]), 6) for i in range(num_classes)
        },
    }
    return weights, report
