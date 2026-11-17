"""TDD tests for causal sequence construction, temporal ordering, stream isolation, and zero leakage."""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest

G2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_DIR = os.path.dirname(os.path.dirname(G2_DIR))
for _p in (V2_DIR, G2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as g2_cfg
import preprocessing as prep
import sequence_builder as sb


def _make_synthetic_df(rows: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a minimal synthetic flow DataFrame + 66-dim feature matrix where feature 0 holds flow_id."""
    records = []
    feats = np.zeros((len(rows), 66), dtype=np.float32)
    for idx, r in enumerate(rows):
        rec = {
            "round": r["round"],
            "src_ip": r["src_ip"],
            "src_port": 12345 + idx,
            "dst_ip": "10.10.10.2",
            "dst_port": 80,
            "bidirectional_first_seen_ms": r["first_ms"],
            "bidirectional_last_seen_ms": r["last_ms"],
            "label": r["label"],
        }
        for c in g2_cfg.FEATURE_COLUMNS:
            rec[c] = 0.0
        rec[g2_cfg.FEATURE_COLUMNS[0]] = float(r["flow_id"])
        feats[idx, 0] = float(r["flow_id"])
        records.append(rec)
    return pd.DataFrame(records), feats


def test_causal_ordering_and_no_future_leakage_on_shuffled_rows():
    # Intentionally shuffled timestamps within one (round, src_ip) stream
    rows = [
        {"round": 3, "src_ip": "10.10.10.1", "first_ms": 300, "last_ms": 310, "label": "scan", "flow_id": 3},
        {"round": 3, "src_ip": "10.10.10.1", "first_ms": 100, "last_ms": 110, "label": "normal", "flow_id": 1},
        {"round": 3, "src_ip": "10.10.10.1", "first_ms": 400, "last_ms": 410, "label": "scan", "flow_id": 4},
        {"round": 3, "src_ip": "10.10.10.1", "first_ms": 200, "last_ms": 210, "label": "bruteforce", "flow_id": 2},
    ]
    df, X_scaled = _make_synthetic_df(rows)
    y_enc = prep.encode_labels(df["label"].to_numpy(), g2_cfg.G2Config().classes)

    seq_data = sb.build_causal_sequences(df, X_scaled, y_encoded=y_enc, seq_len=3)
    X_seq = seq_data["X"]
    lengths = seq_data["lengths"]
    mask = seq_data["mask"]
    y_seq = seq_data["y"]

    assert X_seq.shape == (4, 3, 66)
    assert lengths.tolist() == [1, 2, 3, 3]

    # Sequence 0 (flow_id 1): [1, pad, pad], label=normal
    assert X_seq[0, 0, 0] == 1.0
    assert np.all(X_seq[0, 1:] == 0.0)
    assert mask[0].tolist() == [True, False, False]
    assert y_seq[0] == g2_cfg.G2Config().classes.index("normal")

    # Sequence 1 (flow_id 2): [1, 2, pad], label=bruteforce (must NOT include flow 3 or 4)
    assert X_seq[1, :2, 0].tolist() == [1.0, 2.0]
    assert np.all(X_seq[1, 2:] == 0.0)
    assert mask[1].tolist() == [True, True, False]
    assert y_seq[1] == g2_cfg.G2Config().classes.index("bruteforce")

    # Sequence 2 (flow_id 3): [1, 2, 3], label=scan
    assert X_seq[2, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert mask[2].tolist() == [True, True, True]

    # Sequence 3 (flow_id 4): [2, 3, 4], label=scan
    assert X_seq[3, :, 0].tolist() == [2.0, 3.0, 4.0]


def test_stream_isolation_across_src_ip_and_round_boundaries():
    rows = [
        {"round": 3, "src_ip": "10.10.10.1", "first_ms": 100, "last_ms": 110, "label": "scan", "flow_id": 11},
        {"round": 3, "src_ip": "10.10.10.1", "first_ms": 200, "last_ms": 210, "label": "scan", "flow_id": 12},
        {"round": 3, "src_ip": "10.10.10.3", "first_ms": 150, "last_ms": 160, "label": "normal", "flow_id": 31},
        {"round": 4, "src_ip": "10.10.10.1", "first_ms": 300, "last_ms": 310, "label": "normal", "flow_id": 41},
    ]
    df, X_scaled = _make_synthetic_df(rows)
    seq_data = sb.build_causal_sequences(df, X_scaled, y_encoded=None, seq_len=4)

    # Locate each target flow_id in output sequences
    by_target = {}
    for i in range(len(df)):
        L_eff = int(seq_data["lengths"][i])
        valid_ids = seq_data["X"][i, :L_eff, 0].tolist()
        by_target[int(valid_ids[-1])] = valid_ids

    # 10.10.10.3 (flow 31) must never contain flows from 10.10.10.1
    assert by_target[31] == [31.0]
    # Round 4 flow 41 from 10.10.10.1 must never contain Round 3 flows 11 or 12
    assert by_target[41] == [41.0]
    # Round 3 flow 12 sees only Round 3 same-IP flow 11
    assert by_target[12] == [11.0, 12.0]


def test_real_dataset_sequence_counts_and_zero_train_test_leakage():
    cfg = g2_cfg.G2Config()
    df = prep.load_flows_csv(cfg.data_path)
    tr, te = prep.split_by_round(df, cfg.train_rounds, cfg.test_round)

    p = prep.FlowFeaturePreprocessor().fit(tr)
    X_tr = p.transform(tr)
    X_te = p.transform(te)
    y_tr = prep.encode_labels(tr["label"].to_numpy(), cfg.classes)
    y_te = prep.encode_labels(te["label"].to_numpy(), cfg.classes)

    tr_seq = sb.build_causal_sequences(tr, X_tr, y_encoded=y_tr, seq_len=cfg.seq_len)
    te_seq = sb.build_causal_sequences(te, X_te, y_encoded=y_te, seq_len=cfg.seq_len)

    # 100% of flows preserved for exact 1-to-1 comparison with XGBoost
    assert tr_seq["X"].shape == (7941, cfg.seq_len, 66)
    assert te_seq["X"].shape == (2735, cfg.seq_len, 66)
    assert set(np.unique(tr_seq["rounds"]).tolist()) == {3, 4, 5}
    assert set(np.unique(te_seq["rounds"]).tolist()) == {6}

    # Check padded slots are strictly zero and lengths in [1, seq_len]
    assert int(te_seq["lengths"].min()) >= 1
    assert int(te_seq["lengths"].max()) == cfg.seq_len
    for i in range(len(te_seq["lengths"])):
        L_eff = int(te_seq["lengths"][i])
        if L_eff < cfg.seq_len:
            assert np.all(te_seq["X"][i, L_eff:] == 0.0)
            assert np.all(~te_seq["mask"][i, L_eff:])
        assert np.all(te_seq["mask"][i, :L_eff])


def test_causal_flow_buffer_matches_batch_sequence_builder():
    rows = [
        {"round": 6, "src_ip": "10.10.10.1", "first_ms": 100, "last_ms": 110, "label": "normal", "flow_id": 1},
        {"round": 6, "src_ip": "10.10.10.3", "first_ms": 120, "last_ms": 130, "label": "normal", "flow_id": 2},
        {"round": 6, "src_ip": "10.10.10.1", "first_ms": 200, "last_ms": 210, "label": "scan", "flow_id": 3},
        {"round": 6, "src_ip": "10.10.10.1", "first_ms": 300, "last_ms": 310, "label": "scan", "flow_id": 4},
        {"round": 6, "src_ip": "10.10.10.1", "first_ms": 400, "last_ms": 410, "label": "scan", "flow_id": 5},
    ]
    df, X_scaled = _make_synthetic_df(rows)
    batch = sb.build_causal_sequences(df, X_scaled, y_encoded=None, seq_len=3)

    # Stream in exact arrival order (sorted by last_seen_ms, first_seen_ms)
    buf = sb.CausalFlowBuffer(seq_len=3, feature_dim=66)
    streamed = {}
    for row_idx in range(len(df)):
        src_ip = str(df.iloc[row_idx]["src_ip"])
        x_seq_1, L_eff = buf.push_and_build(src_ip, X_scaled[row_idx])
        assert x_seq_1.shape == (1, 3, 66)
        target_id = int(X_scaled[row_idx, 0])
        streamed[target_id] = (x_seq_1[0], L_eff)

    for i in range(len(df)):
        L_b = int(batch["lengths"][i])
        target_id = int(batch["X"][i, L_b - 1, 0])
        x_s, L_s = streamed[target_id]
        assert L_s == L_b
        np.testing.assert_allclose(x_s, batch["X"][i])
