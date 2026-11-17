"""Training pipeline for G2 Causal LSTM with Masked Temporal Attention.

Strict zero-leakage guarantees:
- Round 6 is NEVER read or touched during preprocessing, CV, hyperparameter/epoch selection,
  class-weight calculation, or final model training.
- Validation uses 3-fold Leave-One-Round-Out CV strictly within training rounds {3, 4, 5},
  refitting the preprocessor and class weights inside each fold.
- Final model is trained on all training rounds {3, 4, 5} for the CV-selected epoch budget
  and saved strictly inside `experiments/g2_sequence_attention/artifacts/`.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as g2_cfg
import metrics as g2_metrics
import model as g2_model
import preprocessing as prep
import sequence_builder as sb


def set_seed(seed: int = 42) -> None:
    """Configure deterministic RNG seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloader(
    X_seq: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X_seq, dtype=np.float32)),
        torch.from_numpy(np.asarray(lengths, dtype=np.int64)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch and return mean cross-entropy loss."""
    net.train()
    total_loss = 0.0
    total_samples = 0

    for xb, lb, yb in loader:
        xb = xb.to(device)
        lb = lb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = net(xb, lb)
        loss = criterion(logits, yb)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        optimizer.step()

        bsz = int(yb.shape[0])
        total_loss += float(loss.item()) * bsz
        total_samples += bsz

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate_loader(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run deterministic inference over a DataLoader -> (pred_indices, probs, true_indices)."""
    net.eval()
    preds_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    for xb, lb, yb in loader:
        xb = xb.to(device)
        lb = lb.to(device)
        logits = net(xb, lb)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        preds_all.append(preds)
        probs_all.append(probs)
        targets_all.append(yb.numpy())

    return (
        np.concatenate(preds_all, axis=0),
        np.concatenate(probs_all, axis=0),
        np.concatenate(targets_all, axis=0),
    )


def build_model_from_config(cfg: g2_cfg.G2Config) -> g2_model.CausalLSTMAttention:
    return g2_model.CausalLSTMAttention(
        input_dim=cfg.input_dim,
        num_classes=len(cfg.classes),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        proj_dropout=cfg.proj_dropout,
        lstm_dropout=cfg.lstm_dropout,
        head_dropout=cfg.head_dropout,
    )


def run_leave_one_round_out_cv(
    train_df: pd.DataFrame,
    cfg: g2_cfg.G2Config,
    device: torch.device,
) -> dict[str, Any]:
    """3-fold Leave-One-Round-Out CV strictly on training rounds {3, 4, 5}.

    Refits preprocessor and class weights inside each fold to guarantee zero validation leakage.
    """
    fold_reports: list[dict[str, Any]] = []
    best_epochs_list: list[int] = []

    for fold_idx, val_round in enumerate(cfg.train_rounds):
        tr_rounds = [r for r in cfg.train_rounds if r != val_round]
        fold_tr_df, fold_val_df = prep.split_by_round(train_df, train_rounds=tr_rounds, test_round=val_round)

        scaler = prep.FlowFeaturePreprocessor().fit(fold_tr_df)
        X_tr = scaler.transform(fold_tr_df)
        X_val = scaler.transform(fold_val_df)
        y_tr = prep.encode_labels(fold_tr_df[g2_cfg.LABEL_COLUMN].to_numpy(), cfg.classes)
        y_val = prep.encode_labels(fold_val_df[g2_cfg.LABEL_COLUMN].to_numpy(), cfg.classes)

        tr_seq = sb.build_causal_sequences(fold_tr_df, X_tr, y_encoded=y_tr, seq_len=cfg.seq_len)
        val_seq = sb.build_causal_sequences(fold_val_df, X_val, y_encoded=y_val, seq_len=cfg.seq_len)

        weights_np, _ = prep.compute_class_weights(tr_seq["y"], cfg.classes)
        set_seed(cfg.seed + fold_idx)

        net = build_model_from_config(cfg).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_np).to(device))
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

        tr_loader = make_dataloader(
            tr_seq["X"], tr_seq["lengths"], tr_seq["y"],
            batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed + fold_idx,
        )
        val_loader = make_dataloader(
            val_seq["X"], val_seq["lengths"], val_seq["y"],
            batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed,
        )

        best_val_f1 = -1.0
        best_val_loss = float("inf")
        best_epoch = 1
        best_metrics: dict[str, Any] = {}
        stale_epochs = 0

        for epoch in range(1, cfg.max_epochs + 1):
            tr_loss = train_one_epoch(net, tr_loader, criterion, optimizer, device, grad_clip=cfg.grad_clip)
            scheduler.step()

            val_preds, val_probs, val_targets = evaluate_loader(net, val_loader, device)
            # Compute validation NLL for tie-breaking when macro-F1 saturates near 1.0
            eps = 1e-12
            val_nll = float(-np.mean(np.log(np.clip(val_probs[np.arange(len(val_targets)), val_targets], eps, 1.0))))
            m = g2_metrics.compute_classification_metrics(
                prep.decode_labels(val_targets, cfg.classes),
                prep.decode_labels(val_preds, cfg.classes),
                classes=cfg.classes,
            )
            improved = (m["macro_f1"] > best_val_f1 + 1e-5) or (
                abs(m["macro_f1"] - best_val_f1) <= 1e-5 and val_nll < best_val_loss - 1e-5
            )
            if improved:
                best_val_f1 = m["macro_f1"]
                best_val_loss = val_nll
                best_epoch = epoch
                best_metrics = {
                    "accuracy": round(m["accuracy"], 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "normal_false_alarm_rate": round(m["normal_false_alarm_rate"], 4),
                    "val_nll": round(val_nll, 6),
                    "train_loss": round(tr_loss, 6),
                    **{f"recall_{c}": round(m[f"recall_{c}"], 4) for c in cfg.classes},
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.patience:
                    break

        best_epochs_list.append(best_epoch)
        fold_reports.append(
            {
                "val_round": int(val_round),
                "train_rounds": [int(r) for r in tr_rounds],
                "best_epoch": int(best_epoch),
                "metrics": best_metrics,
            }
        )
        print(
            f"  [CV Fold val=round {val_round}] best_epoch={best_epoch:2d} | "
            f"macro_f1={best_metrics['macro_f1']:.4f} | acc={best_metrics['accuracy']:.4f} | "
            f"bf_rec={best_metrics['recall_bruteforce']:.4f} | far={best_metrics['normal_false_alarm_rate']:.4f}"
        )

    selected_epochs = max(5, int(round(float(np.mean(best_epochs_list)))))
    mean_macro_f1 = float(np.mean([f["metrics"]["macro_f1"] for f in fold_reports]))
    mean_acc = float(np.mean([f["metrics"]["accuracy"] for f in fold_reports]))
    return {
        "folds": fold_reports,
        "mean_val_macro_f1": round(mean_macro_f1, 4),
        "mean_val_accuracy": round(mean_acc, 4),
        "selected_final_epochs": selected_epochs,
    }


def train_final_model(
    train_df: pd.DataFrame,
    cfg: g2_cfg.G2Config,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    """Fit preprocessor, class weights, and final CausalLSTMAttention on all training rounds {3, 4, 5}."""
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    set_seed(cfg.seed)

    scaler = prep.FlowFeaturePreprocessor().fit(train_df)
    X_tr = scaler.transform(train_df)
    y_tr = prep.encode_labels(train_df[g2_cfg.LABEL_COLUMN].to_numpy(), cfg.classes)
    tr_seq = sb.build_causal_sequences(train_df, X_tr, y_encoded=y_tr, seq_len=cfg.seq_len)

    weights_np, imbalance_report = prep.compute_class_weights(tr_seq["y"], cfg.classes)

    net = build_model_from_config(cfg).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_np).to(device))
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    tr_loader = make_dataloader(
        tr_seq["X"], tr_seq["lengths"], tr_seq["y"],
        batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed,
    )

    history: list[dict[str, float | int]] = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(net, tr_loader, criterion, optimizer, device, grad_clip=cfg.grad_clip)
        scheduler.step()
        history.append(
            {"epoch": epoch, "train_loss": round(loss, 6), "lr": round(float(optimizer.param_groups[0]["lr"]), 8)}
        )
    train_time_s = time.perf_counter() - t0

    ckpt_path = os.path.join(cfg.artifacts_dir, "g2_lstm_attention.pt")
    torch.save(
        {
            "model_state_dict": net.cpu().state_dict(),
            "preprocessor_state": scaler.state_dict(),
            "config": cfg.to_dict(),
            "class_weights": weights_np.tolist(),
            "trained_epochs": epochs,
            "num_parameters": g2_model.count_parameters(net),
        },
        ckpt_path,
    )
    model_size_kb = os.path.getsize(ckpt_path) / 1024.0

    return {
        "checkpoint_path": ckpt_path,
        "trained_epochs": epochs,
        "train_time_seconds": round(train_time_s, 3),
        "num_parameters": g2_model.count_parameters(net),
        "checkpoint_size_kb": round(model_size_kb, 2),
        "imbalance_report": imbalance_report,
        "history": history,
    }


def main() -> None:
    cfg = g2_cfg.G2Config()
    device = select_device(prefer_cuda=True)
    print(f"G2 Training | device={device} | seed={cfg.seed} | seq_len={cfg.seq_len} | features={cfg.input_dim}")

    df = prep.load_flows_csv(cfg.data_path)
    # Strictly isolate training rounds {3, 4, 5}; Round 6 is NEVER used here
    train_df, _ = prep.split_by_round(df, train_rounds=cfg.train_rounds, test_round=cfg.test_round)
    print(f"Loaded training split: {len(train_df)} flows from rounds {list(cfg.train_rounds)}")

    print("\n--- Stage 1: 3-Fold Leave-One-Round-Out CV (Rounds 3, 4, 5 only) ---")
    cv_summary = run_leave_one_round_out_cv(train_df, cfg, device)
    print(
        f"CV Summary: mean_macro_f1={cv_summary['mean_val_macro_f1']:.4f} | "
        f"mean_acc={cv_summary['mean_val_accuracy']:.4f} | "
        f"selected_final_epochs={cv_summary['selected_final_epochs']}"
    )

    print("\n--- Stage 2: Final Model Training (Rounds 3, 4, 5) ---")
    final_summary = train_final_model(train_df, cfg, epochs=cv_summary["selected_final_epochs"], device=device)
    print(
        f"Saved checkpoint: {final_summary['checkpoint_path']} "
        f"({final_summary['num_parameters']:,} params, {final_summary['checkpoint_size_kb']:.1f} KB, "
        f"{final_summary['train_time_seconds']:.2f}s)"
    )

    report = {
        "device_used": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "config": cfg.to_dict(),
        "cv_summary": cv_summary,
        "final_training": final_summary,
    }
    report_path = os.path.join(cfg.artifacts_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved training report: {report_path}")


if __name__ == "__main__":
    main()
