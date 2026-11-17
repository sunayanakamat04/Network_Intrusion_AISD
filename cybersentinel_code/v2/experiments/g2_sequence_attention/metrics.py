"""Evaluation metrics, normal false-alarm rate, latency benchmarking, and comparison tables."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import time
from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    classes: Sequence[str],
    normal_label: str = "normal",
) -> dict[str, Any]:
    """Compute required multiclass and IDS-specific metrics using exact dataset class names.

    False-alarm rate on normal traffic =
        (normal flows misclassified as any attack) / (total actual normal flows).
    """
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    cls_list = list(classes)

    acc = float(accuracy_score(yt, yp))
    macro_f1 = float(f1_score(yt, yp, labels=cls_list, average="macro", zero_division=0))
    per_class_recall = recall_score(yt, yp, labels=cls_list, average=None, zero_division=0)

    is_normal_true = yt == normal_label
    is_attack_pred = yp != normal_label
    n_normal = int(is_normal_true.sum())
    n_attack = int((~is_normal_true).sum())

    fp_normal = int((is_normal_true & is_attack_pred).sum())
    tp_attack = int(((~is_normal_true) & is_attack_pred).sum())

    normal_far = float(fp_normal / n_normal) if n_normal > 0 else float("nan")
    attack_recall = float(tp_attack / n_attack) if n_attack > 0 else float("nan")

    cm = confusion_matrix(yt, yp, labels=cls_list)
    report_dict = classification_report(
        yt, yp, labels=cls_list, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        yt, yp, labels=cls_list, digits=4, zero_division=0
    )

    result: dict[str, Any] = {
        "n_samples": int(len(yt)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "attack_recall": attack_recall,
        "normal_false_alarm_rate": normal_far,
        "normal_false_positives": fp_normal,
        "total_normal_flows": n_normal,
        "classes": cls_list,
        "confusion_matrix": cm.tolist(),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }
    for cls_name, rec_val in zip(cls_list, per_class_recall):
        result[f"recall_{cls_name}"] = float(rec_val)
    return result


def measure_single_flow_latency_ms(
    predict_fn: Callable[[Any], Any],
    items: Sequence[Any],
    warmup: int = 20,
    sync_fn: Callable[[], None] | None = None,
) -> dict[str, float | int]:
    """Measure actual wall-clock single-flow prediction latency in milliseconds.

    Args:
        predict_fn: Callable accepting a single flow item and returning a prediction.
        items: Sequence of single-flow inputs (processed one at a time, never batched).
        warmup: Number of leading items used for warm-up (excluded from timing stats).
        sync_fn: Optional device synchronization callback (e.g. `torch.cuda.synchronize`).
    """
    n_total = len(items)
    if n_total == 0:
        raise ValueError("Cannot measure latency on an empty item sequence.")
    n_warm = min(max(0, warmup), max(0, n_total - 1))

    for i in range(n_warm):
        predict_fn(items[i])
    if sync_fn is not None:
        sync_fn()

    times_ms: list[float] = []
    for i in range(n_warm, n_total):
        if sync_fn is not None:
            sync_fn()
        t0 = time.perf_counter()
        predict_fn(items[i])
        if sync_fn is not None:
            sync_fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "n_measured": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def format_comparison_markdown(
    model_rows: Sequence[dict[str, Any]],
    classes: Sequence[str],
) -> str:
    """Format a Markdown comparison table across baseline and G2 models."""
    headers = ["Metric"] + [str(r["model_name"]) for r in model_rows]
    sep = ["---"] * len(headers)

    metric_specs: list[tuple[str, str, str]] = [
        ("Accuracy", "accuracy", "{:.4f}"),
        ("Macro-F1", "macro_f1", "{:.4f}"),
    ]
    for c in classes:
        metric_specs.append((f"{c} Recall", f"recall_{c}", "{:.4f}"))
    metric_specs.extend(
        [
            ("False Alarm Rate (normal)", "normal_false_alarm_rate", "{:.4f}"),
            ("Prediction Time (ms/flow)", "predict_ms_mean", "{:.4f}"),
            ("Model Size (KB)", "model_size_kb", "{:.1f}"),
        ]
    )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for label, key, fmt in metric_specs:
        vals = [fmt.format(r[key]) if key in r and r[key] is not None else "N/A" for r in model_rows]
        lines.append("| " + " | ".join([label] + vals) + " |")
    return "\n".join(lines)
