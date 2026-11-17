"""Fair Round 6 evaluation comparing G2 Causal LSTM-Attention against both XGBoost baselines.

Compares on identical held-out Round 6 (2,735 flows):
1. Shipped XGBoost Baseline (22 selected features, read-only from `v2/models/best_xgb.json`).
2. XGBoost Baseline (66 features, trained separately on Rounds 3-5 with identical hyperparams).
3. G2 Causal LSTM-Attention (66 features, causal window L=8, trained strictly on Rounds 3-5).

Measures actual wall-clock single-flow prediction latency for every model, including full
per-flow preprocessing and stateful causal history buffer handling for G2.
"""
from __future__ import annotations

import json
import os
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

import config as g2_cfg
import metrics as g2_metrics
import model as g2_model
import preprocessing as prep
import sequence_builder as sb
import train as g2_train


def _sort_round_arrival_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort flows in causal arrival order (last_seen_ms, first_seen_ms) for realistic streaming."""
    return (
        df.assign(_orig_pos=np.arange(len(df)))
        .sort_values(
            by=[g2_cfg.ORDER_COLUMNS[0], g2_cfg.ORDER_COLUMNS[1], "_orig_pos"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .drop(columns=["_orig_pos"])
        .reset_index(drop=True)
    )


def evaluate_shipped_xgb_22(
    test_df: pd.DataFrame,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate the untouched shipped Pipeline v2 XGBoost model (22 selected features)."""
    with open(g2_cfg.DEFAULT_SHIPPED_XGB_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feats_22: list[str] = list(meta["features"])
    model_classes: list[str] = list(meta["classes"])

    xgb_22 = XGBClassifier()
    xgb_22.load_model(g2_cfg.DEFAULT_SHIPPED_XGB_MODEL)

    X_te = test_df[feats_22].to_numpy(dtype=np.float32)
    y_te = test_df[g2_cfg.LABEL_COLUMN].to_numpy()
    raw_preds = xgb_22.predict(X_te)
    y_pred = np.array([model_classes[int(i)] for i in raw_preds], dtype=object)

    m = g2_metrics.compute_classification_metrics(y_te, y_pred, classes=classes)

    # Measure actual single-flow latency on 500 single-row NumPy calls (matching Pipeline v2)
    sample_rows = [X_te[i : i + 1] for i in range(min(520, len(X_te)))]
    lat = g2_metrics.measure_single_flow_latency_ms(lambda r: xgb_22.predict(r), sample_rows, warmup=20)
    size_kb = os.path.getsize(g2_cfg.DEFAULT_SHIPPED_XGB_MODEL) / 1024.0

    return {
        "model_name": "XGBoost Baseline (22 feat, shipped)",
        "num_features": len(feats_22),
        "metrics": m,
        "latency_cpu": lat,
        "predict_ms_mean": lat["mean_ms"],
        "model_size_kb": round(size_kb, 2),
        "y_true": y_te,
        "y_pred": y_pred,
    }


def evaluate_xgb_66_reference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classes: tuple[str, ...],
    artifacts_dir: str,
) -> dict[str, Any]:
    """Train and evaluate a separate 66-feature XGBoost reference (same hyperparams as train_models.py)."""
    feats_66 = list(g2_cfg.FEATURE_COLUMNS)
    prep.validate_feature_columns(feats_66)

    X_tr = train_df[feats_66].to_numpy(dtype=np.float32)
    y_tr_idx = prep.encode_labels(train_df[g2_cfg.LABEL_COLUMN].to_numpy(), classes)
    X_te = test_df[feats_66].to_numpy(dtype=np.float32)
    y_te = test_df[g2_cfg.LABEL_COLUMN].to_numpy()

    xgb_66 = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
        random_state=0,
    )
    xgb_66.fit(X_tr, y_tr_idx)

    os.makedirs(artifacts_dir, exist_ok=True)
    ref_path = os.path.join(artifacts_dir, "xgb_66_reference.json")
    xgb_66.save_model(ref_path)

    raw_preds = xgb_66.predict(X_te)
    y_pred = prep.decode_labels(raw_preds, classes)
    m = g2_metrics.compute_classification_metrics(y_te, y_pred, classes=classes)

    sample_rows = [X_te[i : i + 1] for i in range(min(520, len(X_te)))]
    lat = g2_metrics.measure_single_flow_latency_ms(lambda r: xgb_66.predict(r), sample_rows, warmup=20)
    size_kb = os.path.getsize(ref_path) / 1024.0

    return {
        "model_name": "XGBoost Baseline (66 feat)",
        "num_features": len(feats_66),
        "metrics": m,
        "latency_cpu": lat,
        "predict_ms_mean": lat["mean_ms"],
        "model_size_kb": round(size_kb, 2),
        "y_true": y_te,
        "y_pred": y_pred,
    }


def _benchmark_g2_streaming(
    net: g2_model.CausalLSTMAttention,
    scaler: prep.FlowFeaturePreprocessor,
    stream_df: pd.DataFrame,
    seq_len: int,
    classes: tuple[str, ...],
    device: torch.device,
    max_flows: int = 520,
    warmup: int = 20,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Measure ACTUAL single-flow causal latency (end-to-end with history buffer + model-only)."""
    net_dev = net.to(device).eval()
    sync_fn = torch.cuda.synchronize if device.type == "cuda" else None

    raw_feats = stream_df.loc[:, list(g2_cfg.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    src_ips = stream_df[g2_cfg.GROUP_COLUMN].astype(str).to_numpy()
    n_items = min(max_flows, len(stream_df))
    stream_items = [(src_ips[i], raw_feats[i]) for i in range(n_items)]

    # 1. Full end-to-end single-flow causal inference (preprocess + causal history buffer + forward)
    buf_e2e = sb.CausalFlowBuffer(seq_len=seq_len, feature_dim=len(g2_cfg.FEATURE_COLUMNS))

    @torch.no_grad()
    def _predict_e2e(item: tuple[str, np.ndarray]) -> int:
        src_ip, raw_vec = item
        scaled_vec = scaler.transform_single(raw_vec)
        x_np, eff_len = buf_e2e.push_and_build(src_ip, scaled_vec)
        xt = torch.from_numpy(x_np).to(device)
        lt = torch.tensor([eff_len], dtype=torch.long, device=device)
        logits = net_dev(xt, lt)
        return int(torch.argmax(logits, dim=-1).item())

    e2e_lat = g2_metrics.measure_single_flow_latency_ms(
        _predict_e2e, stream_items, warmup=warmup, sync_fn=sync_fn
    )

    # 2. Model-forward-only latency (for transparent component breakdown)
    buf_pre = sb.CausalFlowBuffer(seq_len=seq_len, feature_dim=len(g2_cfg.FEATURE_COLUMNS))
    prebuilt_tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
    for src_ip, raw_vec in stream_items:
        scaled_vec = scaler.transform_single(raw_vec)
        x_np, eff_len = buf_pre.push_and_build(src_ip, scaled_vec)
        prebuilt_tensors.append(
            (torch.from_numpy(x_np).to(device), torch.tensor([eff_len], dtype=torch.long, device=device))
        )

    @torch.no_grad()
    def _predict_model_only(pair: tuple[torch.Tensor, torch.Tensor]) -> int:
        xt, lt = pair
        logits = net_dev(xt, lt)
        return int(torch.argmax(logits, dim=-1).item())

    model_lat = g2_metrics.measure_single_flow_latency_ms(
        _predict_model_only, prebuilt_tensors, warmup=warmup, sync_fn=sync_fn
    )
    return e2e_lat, model_lat


@torch.no_grad()
def evaluate_g2_model(
    test_df: pd.DataFrame,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Evaluate trained G2 Causal LSTM-Attention on all 2,735 held-out Round 6 flows."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = g2_cfg.G2Config.from_dict(ckpt["config"])
    scaler = prep.FlowFeaturePreprocessor.from_state_dict(ckpt["preprocessor_state"])

    net = g2_train.build_model_from_config(cfg)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Batch causal evaluation over all 2,735 Round 6 flows
    X_te_scaled = scaler.transform(test_df)
    y_te_enc = prep.encode_labels(test_df[g2_cfg.LABEL_COLUMN].to_numpy(), cfg.classes)
    te_seq = sb.build_causal_sequences(test_df, X_te_scaled, y_encoded=y_te_enc, seq_len=cfg.seq_len)

    eval_device = g2_train.select_device(prefer_cuda=True)
    net_eval = net.to(eval_device)
    te_loader = g2_train.make_dataloader(
        te_seq["X"], te_seq["lengths"], te_seq["y"],
        batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed,
    )
    pred_idx, _, true_idx = g2_train.evaluate_loader(net_eval, te_loader, eval_device)
    y_true_str = prep.decode_labels(true_idx, cfg.classes)
    y_pred_str = prep.decode_labels(pred_idx, cfg.classes)

    # Verify streaming CausalFlowBuffer matches batch causal predictions 100% on Round 6
    stream_df = _sort_round_arrival_order(test_df)
    buf_verify = sb.CausalFlowBuffer(seq_len=cfg.seq_len, feature_dim=len(g2_cfg.FEATURE_COLUMNS))
    raw_stream = stream_df.loc[:, list(g2_cfg.FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    src_stream = stream_df[g2_cfg.GROUP_COLUMN].astype(str).to_numpy()
    stream_preds: list[int] = []
    net_cpu = net.to(torch.device("cpu")).eval()
    for i in range(len(stream_df)):
        sv = scaler.transform_single(raw_stream[i])
        x_np, eff_len = buf_verify.push_and_build(src_stream[i], sv)
        logits = net_cpu(torch.from_numpy(x_np), torch.tensor([eff_len], dtype=torch.long))
        stream_preds.append(int(torch.argmax(logits, dim=-1).item()))
    y_stream_true = stream_df[g2_cfg.LABEL_COLUMN].to_numpy()
    y_stream_pred = prep.decode_labels(np.asarray(stream_preds, dtype=np.int64), cfg.classes)

    m_batch = g2_metrics.compute_classification_metrics(y_true_str, y_pred_str, classes=cfg.classes)
    m_stream = g2_metrics.compute_classification_metrics(y_stream_true, y_stream_pred, classes=cfg.classes)
    streaming_matches_batch = (
        m_batch["confusion_matrix"] == m_stream["confusion_matrix"]
    )

    # Measure ACTUAL single-flow causal latency on CPU (fair comparison to CPU XGBoost / live IDS)
    e2e_cpu, model_cpu = _benchmark_g2_streaming(
        net, scaler, stream_df, cfg.seq_len, cfg.classes, device=torch.device("cpu")
    )

    # Also measure GPU single-flow latency if CUDA is available
    e2e_gpu = None
    model_gpu = None
    if torch.cuda.is_available():
        e2e_gpu, model_gpu = _benchmark_g2_streaming(
            net, scaler, stream_df, cfg.seq_len, cfg.classes, device=torch.device("cuda")
        )

    size_kb = os.path.getsize(checkpoint_path) / 1024.0
    return {
        "model_name": f"G2 Causal LSTM-Attention (66 feat, L={cfg.seq_len})",
        "num_features": cfg.input_dim,
        "seq_len": cfg.seq_len,
        "num_parameters": int(ckpt["num_parameters"]),
        "streaming_matches_batch": streaming_matches_batch,
        "metrics": m_batch,
        "latency_cpu_e2e": e2e_cpu,
        "latency_cpu_model_only": model_cpu,
        "latency_gpu_e2e": e2e_gpu,
        "latency_gpu_model_only": model_gpu,
        "predict_ms_mean": e2e_cpu["mean_ms"],
        "model_size_kb": round(size_kb, 2),
        "y_true": y_true_str,
        "y_pred": y_pred_str,
    }


def save_confusion_matrix_plot(
    results: list[dict[str, Any]],
    classes: tuple[str, ...],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.6))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = np.array(res["metrics"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(res["model_name"], fontsize=10)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Held-Out Round 6 Confusion Matrices (2,735 flows)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_full_evaluation(cfg: g2_cfg.G2Config | None = None) -> dict[str, Any]:
    cfg = cfg or g2_cfg.G2Config()
    df = prep.load_flows_csv(cfg.data_path)
    train_df, test_df = prep.split_by_round(df, train_rounds=cfg.train_rounds, test_round=cfg.test_round)

    ckpt_path = os.path.join(cfg.artifacts_dir, "g2_lstm_attention.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"G2 checkpoint not found at {ckpt_path}. Run train.py first.")

    res_xgb22 = evaluate_shipped_xgb_22(test_df, cfg.classes)
    res_xgb66 = evaluate_xgb_66_reference(train_df, test_df, cfg.classes, cfg.artifacts_dir)
    res_g2 = evaluate_g2_model(test_df, ckpt_path)

    all_res = [res_xgb22, res_xgb66, res_g2]
    table_rows: list[dict[str, Any]] = []
    for r in all_res:
        m = r["metrics"]
        table_rows.append(
            {
                "model_name": r["model_name"],
                "num_features": r["num_features"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                **{f"recall_{c}": m[f"recall_{c}"] for c in cfg.classes},
                "normal_false_alarm_rate": m["normal_false_alarm_rate"],
                "attack_recall": m["attack_recall"],
                "predict_ms_mean": r["predict_ms_mean"],
                "model_size_kb": r["model_size_kb"],
            }
        )

    csv_path = os.path.join(cfg.artifacts_dir, "comparison_table.csv")
    pd.DataFrame(table_rows).to_csv(csv_path, index=False)

    cm_path = os.path.join(cfg.artifacts_dir, "confusion_matrices.png")
    save_confusion_matrix_plot(all_res, cfg.classes, cm_path)

    # Strip large raw prediction arrays before JSON dump
    json_payload = {
        "test_round": cfg.test_round,
        "test_flows": int(len(test_df)),
        "classes": list(cfg.classes),
        "comparison_table": table_rows,
        "models": [
            {k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
            for r in all_res
        ],
    }
    json_path = os.path.join(cfg.artifacts_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    md_table = g2_metrics.format_comparison_markdown(table_rows, cfg.classes)
    print("\n=== HELD-OUT ROUND 6 FAIR COMPARISON (2,735 flows) ===")
    print(md_table)
    print(f"\nG2 streaming-vs-batch exact match: {res_g2['streaming_matches_batch']}")
    print(
        f"G2 CPU single-flow latency: e2e mean={res_g2['latency_cpu_e2e']['mean_ms']:.4f} ms "
        f"(p95={res_g2['latency_cpu_e2e']['p95_ms']:.4f} ms) | "
        f"model-only mean={res_g2['latency_cpu_model_only']['mean_ms']:.4f} ms"
    )
    if res_g2["latency_gpu_e2e"] is not None:
        print(
            f"G2 GPU single-flow latency: e2e mean={res_g2['latency_gpu_e2e']['mean_ms']:.4f} ms "
            f"(p95={res_g2['latency_gpu_e2e']['p95_ms']:.4f} ms) | "
            f"model-only mean={res_g2['latency_gpu_model_only']['mean_ms']:.4f} ms"
        )
    print(f"\nSaved: {csv_path}, {json_path}, {cm_path}")
    return json_payload


if __name__ == "__main__":
    run_full_evaluation()
