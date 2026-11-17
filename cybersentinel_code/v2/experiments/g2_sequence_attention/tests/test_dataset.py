"""TDD tests for G2 dataset loading, feature validation, round splitting, preprocessing, and class weights."""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Allow importing G2 modules and Pipeline v2 features_v2 without modifying sys.path globally
G2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_DIR = os.path.dirname(os.path.dirname(G2_DIR))
for _p in (V2_DIR, G2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import features_v2 as fv
import config as g2_cfg
import preprocessing as prep


def test_config_features_match_v2_and_exclude_metadata():
    assert len(g2_cfg.FEATURE_COLUMNS) == 66
    assert list(g2_cfg.FEATURE_COLUMNS) == list(fv.FEATURES)
    forbidden = set(g2_cfg.ENVELOPE_COLUMNS) | {g2_cfg.ROUND_COLUMN, g2_cfg.LABEL_COLUMN}
    assert forbidden.isdisjoint(set(g2_cfg.FEATURE_COLUMNS))
    assert "src_ip" in g2_cfg.ENVELOPE_COLUMNS
    assert "dst_ip" in g2_cfg.ENVELOPE_COLUMNS
    assert "src_port" in g2_cfg.ENVELOPE_COLUMNS
    assert "dst_port" in g2_cfg.ENVELOPE_COLUMNS
    assert "bidirectional_first_seen_ms" in g2_cfg.ENVELOPE_COLUMNS
    assert "bidirectional_last_seen_ms" in g2_cfg.ENVELOPE_COLUMNS


def test_validate_feature_columns_rejects_metadata_and_wrong_counts():
    prep.validate_feature_columns(g2_cfg.FEATURE_COLUMNS)

    for forbidden_col in [
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "bidirectional_first_seen_ms",
        "bidirectional_last_seen_ms",
        "round",
        "label",
    ]:
        bad = list(g2_cfg.FEATURE_COLUMNS[:-1]) + [forbidden_col]
        with pytest.raises(ValueError):
            prep.validate_feature_columns(bad)

    with pytest.raises(ValueError):
        prep.validate_feature_columns(g2_cfg.FEATURE_COLUMNS[:22])


def test_load_flows_csv_schema_and_counts():
    cfg = g2_cfg.G2Config()
    df = prep.load_flows_csv(cfg.data_path)
    assert df.shape == (10676, 74)
    assert int(df.isna().sum().sum()) == 0
    assert sorted(df["label"].unique().tolist()) == list(cfg.classes)
    assert sorted(df["round"].unique().tolist()) == [3, 4, 5, 6]


def test_round_based_split_is_strict_and_leak_free():
    cfg = g2_cfg.G2Config()
    df = prep.load_flows_csv(cfg.data_path)
    tr, te = prep.split_by_round(df, train_rounds=cfg.train_rounds, test_round=cfg.test_round)

    assert len(tr) == 7941
    assert len(te) == 2735
    assert len(tr) + len(te) == len(df)
    assert set(tr["round"].unique()) == {3, 4, 5}
    assert set(te["round"].unique()) == {6}
    assert set(tr["round"].unique()).isdisjoint(set(te["round"].unique()))

    # Must reject overlapping train/test rounds
    with pytest.raises(ValueError):
        prep.split_by_round(df, train_rounds=(3, 4, 6), test_round=6)


def test_preprocessor_fits_strictly_on_train_and_handles_constant_columns():
    cfg = g2_cfg.G2Config()
    df = prep.load_flows_csv(cfg.data_path)
    tr, te = prep.split_by_round(df, cfg.train_rounds, cfg.test_round)

    # Fit preprocessor on untouched train split
    p1 = prep.FlowFeaturePreprocessor().fit(tr)

    # Mutate test set wildly and verify fitting on train produces identical stats
    te_corrupted = te.copy()
    te_corrupted[g2_cfg.FEATURE_COLUMNS[0]] = 1e12
    p2 = prep.FlowFeaturePreprocessor().fit(tr)
    np.testing.assert_allclose(p1.mean_, p2.mean_)
    np.testing.assert_allclose(p1.std_, p2.std_)

    X_tr = p1.transform(tr)
    X_te = p1.transform(te)
    assert X_tr.shape == (len(tr), 66)
    assert X_te.shape == (len(te), 66)
    assert X_tr.dtype == np.float32
    assert X_te.dtype == np.float32
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_te).all()

    # Modifying src_ip or timestamps must not affect transformed 66-feature matrix
    te_diff_meta = te.copy()
    te_diff_meta["src_ip"] = "192.168.99.99"
    te_diff_meta["dst_port"] = 65535
    te_diff_meta["bidirectional_first_seen_ms"] = 0
    X_te_diff = p1.transform(te_diff_meta)
    np.testing.assert_allclose(X_te, X_te_diff)

    # Single-flow transform must match batch transform
    row_dict = {k: float(te.iloc[0][k]) for k in g2_cfg.FEATURE_COLUMNS}
    single_vec = p1.transform_single(row_dict)
    assert single_vec.shape == (66,)
    assert single_vec.dtype == np.float32
    np.testing.assert_allclose(single_vec, X_te[0], atol=1e-6)

    # State dict roundtrip
    restored = prep.FlowFeaturePreprocessor.from_state_dict(p1.state_dict())
    np.testing.assert_allclose(restored.transform(te[:10]), X_te[:10])


def test_class_weights_handle_bruteforce_imbalance_on_train_only():
    cfg = g2_cfg.G2Config()
    df = prep.load_flows_csv(cfg.data_path)
    tr, _ = prep.split_by_round(df, cfg.train_rounds, cfg.test_round)
    y_tr = prep.encode_labels(tr["label"].to_numpy(), cfg.classes)

    weights, report = prep.compute_class_weights(y_tr, cfg.classes)
    assert weights.shape == (4,)
    assert np.all(weights > 0)

    bf_idx = cfg.classes.index("bruteforce")
    syn_idx = cfg.classes.index("synflood")
    # Bruteforce has only 100 train flows vs 3816 synflood -> much higher weight
    assert report["raw_counts"]["bruteforce"] == 100
    assert report["raw_counts"]["synflood"] == 3816
    assert weights[bf_idx] > weights[syn_idx] * 20.0
    # After inverse-frequency weighting, effective weighted proportions are balanced (~0.25 each)
    for cls in cfg.classes:
        assert abs(report["effective_weighted_proportions"][cls] - 0.25) < 1e-6
