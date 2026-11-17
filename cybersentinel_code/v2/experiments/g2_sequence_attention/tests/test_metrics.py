"""TDD tests for evaluation metrics, normal false-alarm rate, confusion matrix, and latency measurement."""
from __future__ import annotations

import os
import sys
import numpy as np
import pytest

G2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if G2_DIR not in sys.path:
    sys.path.insert(0, G2_DIR)

import metrics as g2_metrics

CLASSES = ("bruteforce", "normal", "scan", "synflood")


def test_classification_metrics_and_normal_false_alarm_rate():
    # 4 normal flows (1 wrongly flagged as scan -> FAR = 1/4 = 0.25)
    # 2 bruteforce (1 missed as normal -> recall = 0.5)
    # 2 scan (both right -> recall = 1.0)
    # 2 synflood (both right -> recall = 1.0)
    y_true = np.array([
        "normal", "normal", "normal", "normal",
        "bruteforce", "bruteforce",
        "scan", "scan",
        "synflood", "synflood",
    ])
    y_pred = np.array([
        "normal", "normal", "normal", "scan",
        "bruteforce", "normal",
        "scan", "scan",
        "synflood", "synflood",
    ])

    res = g2_metrics.compute_classification_metrics(y_true, y_pred, classes=CLASSES)

    assert pytest.approx(res["accuracy"], abs=1e-6) == 0.8
    assert pytest.approx(res["recall_normal"], abs=1e-6) == 0.75
    assert pytest.approx(res["recall_bruteforce"], abs=1e-6) == 0.5
    assert pytest.approx(res["recall_scan"], abs=1e-6) == 1.0
    assert pytest.approx(res["recall_synflood"], abs=1e-6) == 1.0
    # FAR = FP_normal / N_normal = 1 / 4 = 0.25
    assert pytest.approx(res["normal_false_alarm_rate"], abs=1e-6) == 0.25
    assert 0.0 < res["macro_f1"] < 1.0

    cm = np.array(res["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert int(cm.sum()) == len(y_true)
    # Row 1 = normal -> 3 normal, 1 scan
    assert cm[1, 1] == 3 and cm[1, 2] == 1


def test_perfect_predictions_have_zero_false_alarm_rate():
    y = np.array(["bruteforce", "normal", "scan", "synflood"] * 5)
    res = g2_metrics.compute_classification_metrics(y, y, classes=CLASSES)
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["normal_false_alarm_rate"] == 0.0


def test_single_flow_latency_benchmark_measures_real_wall_time():
    items = [np.ones((66,), dtype=np.float32) * i for i in range(50)]
    lat = g2_metrics.measure_single_flow_latency_ms(lambda x: float(x.sum()), items, warmup=10)

    assert lat["n_measured"] == 40
    assert lat["mean_ms"] > 0.0
    assert lat["median_ms"] > 0.0
    assert lat["p95_ms"] >= lat["median_ms"]
