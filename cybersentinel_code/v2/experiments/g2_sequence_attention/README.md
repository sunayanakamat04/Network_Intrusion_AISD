# CyberSentinel AI — G2: Sequence / Attention Deep Learning (Causal LSTM with Masked Temporal Attention)

> **Track:** G2 — Sequence / Attention Deep Learning  
> **Branch:** `feature/g2-sequence-attention`  
> **Scope:** Isolated deep-learning methodology extending CyberSentinel AI Pipeline v2 without modifying teammate-owned pipeline code, the dataset, or the shipped XGBoost baseline.

---

## 1. Introduction

CyberSentinel AI is a live Network Intrusion Detection System (IDS) built on a controlled two-VM lab (`labnet`) using NFStream flow extraction and a verified Pipeline v2 XGBoost baseline (`models/best_xgb.json`). As part of Team 10's comparative methodology study (outlined in [README.md](../../../../README.md) and [GROUPING_ANALYSIS.md](../../../../research_md/GROUPING_ANALYSIS.md)), Track **G2** investigates whether a **sequence- and attention-based deep learning architecture** can detect network intrusions accurately and efficiently on CyberSentinel's own NFStream flow dataset while remaining fast enough for live IDS deployment.

This module implements, tests (via strict SDLC + TDD), trains, and fairly evaluates a **Causal LSTM with Masked Temporal Attention (`CausalLSTMAttention`)** using PyTorch on the exact shared 10,676-flow dataset and round-based split, comparing it head-to-head against both the untouched shipped 22-feature XGBoost baseline and a separately trained 66-feature XGBoost reference.

---

## 2. Problem Statement

Given completed NFStream network flows containing **66 numeric behavioral features** (excluding IP addresses, ports, and timestamps as model inputs), classify each flow into one of four classes observed in CyberSentinel AI:

- `normal`
- `scan` (port scan via `nmap`)
- `synflood` (SYN flood DoS via `hping3`)
- `bruteforce` (SSH password guessing via `hydra` — severe minority class with only 133 total flows)

Key engineering and scientific challenges:
1. **Strict temporal causality and zero leakage:** Raw CSV rows are emitted by NFStream on flow expiration (not monotonically ordered), and Round 6 must remain completely untouched until final evaluation.
2. **Severe class imbalance:** `bruteforce` represents only **1.26%** of training flows (100 out of 7,941).
3. **Padding integrity:** Short/cold-start streams (`< L` flows) must be handled without letting zero-padded timesteps distort LSTM hidden states or attention weights.
4. **Live IDS latency budget:** Accuracy alone is insufficient; single-flow prediction time (including causal history buffering) and normal-traffic false-alarm rate (FAR) must be measured honestly alongside XGBoost.

---

## 3. G2 Research Papers Studied

All three papers in `research_md/G2_Sequence_Attention_DL/` were studied in full:

1. **Berhanu et al. (2025)** — *A recurrent neural network for network-based intrusion detection*, PeerJ Computer Science 11:e3306 (`04_Berhanu_2025_PeerJ_CS_RNN_NIDS.md`).
2. **Zhu et al. (2025)** — *TransNeSt: A Hybrid Split-Attention and Transformer Architecture for High-Performance Network Intrusion Detection*, CMES 145(3) (`03_Zhu_2025_CMES_TransNeSt_NIDS.md`).
3. **Çavşı Zaim & Yolaçan (2025)** — *FPE–Transformer: A Feature Positional Encoding-Based Transformer Model for Attack Detection*, Applied Sciences 15(3):1252 (`11_Zaim_2025_Appl_Sci_FPE_Transformer.md`).

---

## 4. Literature Review

| Paper | Core Architecture | Datasets & Task | Reported Highlights | Key Limitations & Lessons Applied to G2 |
|---|---|---|---|---|
| **Berhanu et al. (2025)** | Stacked LSTM-RNN (80 units/layer, dropout 0.3, Adam, BPTT) | CIC-IDS-2018 & NSL-KDD (binary) | 99.94% acc / 99.95% F1 / 0.036% FAR (CIC-IDS-2018); 99.42% acc (NSL-KDD) | Binary-only (multiclass left for future work); applied SMOTE *before* the random 80/20 split (data leakage). Explicitly recommends keeping layers, hidden units, and sequence length compact to control latency. |
| **Zhu et al. (2025) — TransNeSt** | 1D ResNeSt split-attention (4 blocks, 32 ch, `k=2, r=2`) + 4-layer Transformer Encoder (4 heads, FFN 256, RMSNorm, `[CLS]` head; 0.78M params) | NSL-KDD, UNSW-NB15, CIC-IDS2017, CICIoT2023 (multiclass) | Micro-F1: 99.04% / 91.92% / 99.18% / 97.85%; WGAN outperformed SMOTE on rare minority classes (`Bot`, `BruteForce`, `U2R`) | High computational complexity of split-attention (~0.78M params; standalone Transformer 0.523M) hinders lightweight real-time edge deployment; black-box interpretability. |
| **Zaim & Yolaçan (2025) — FPE-Transformer** | Lightweight 1-layer, 2-head Transformer Encoder + Feature Positional Encoding (FPE) + multi-layer `ClassificationHead` | NSL-KDD & CIC-IDS2017 (binary + multiclass) | Outperformed standard sinusoidal PE and Uniform PE (91.7% binary / 90.6% multiclass on CIC-IDS2017) | Evaluated on only 4 RF-selected features for CIC-IDS2017; omitted macro-F1, per-class recall, normal FAR, and single-flow latency. |

**Synthesis:** Across G2 literature, recurrent models excel at compact causal temporal modeling on modest dataset sizes, while attention mechanisms and dedicated classification heads improve context weighting and non-linear class separation. Crucially, CyberSentinel's own [GROUPING_ANALYSIS.md](../../../../research_md/GROUPING_ANALYSIS.md) notes that public-dataset headline numbers are not transferable across feature extractors and that every model must be judged on the same honest axes: macro-F1, minority-class recall, normal FAR, single-flow latency, and model size.

---

## 5. Selected Methodology

We selected **Option A (Compact Causal LSTM)** enhanced with **Current-Step-Anchored Masked Temporal Attention** (`CausalLSTMAttention`) rather than a heavy standalone Transformer:

1. **Data-regime suitability:** Training rounds `{3, 4, 5}` contain 7,941 flows and only 100 `bruteforce` flows. An LSTM's explicit recurrent inductive bias generalizes cleanly on small-to-medium sequential flow datasets with ~88K parameters (~354 KB), whereas multi-layer Transformers (~523K–780K parameters in Zhu et al.) are over-parameterized for this sample count.
2. **Real-time IDS latency:** A compact unidirectional 2-layer LSTM over causal length `L = 8` keeps CPU single-flow inference well under Pipeline v2's 5 ms/flow budget.
3. **Direct literature synthesis:** Directly extends Berhanu et al. (2025)'s LSTM-RNN to **multiclass classification** with **strict leak-free round splitting** and **class-weighted loss**, while incorporating **masked temporal attention** and a multi-layer **ClassificationHead** inspired by Zhu et al. (2025) and Zaim & Yolaçan (2025).

---

## 6. Architecture Explanation

Implemented in [model.py](model.py) (`CausalLSTMAttention`, **87,812 trainable parameters**):

```
Input X_seq: (B, L=8, 66)  +  lengths: (B,) in [1..8]
   │
   ▼
[1] Input Projection: Linear(66 → 64) → LayerNorm(64) → GELU → Dropout(0.2)
    + Explicit valid_mask multiplication (zeros padded slots)
   │
   ▼
[2] Sequence Packing: pack_padded_sequence(proj, lengths, enforce_sorted=False)
    → 2-layer Unidirectional Causal LSTM(64 → 64, dropout=0.3, bidirectional=False)
    → pad_packed_sequence(total_length=L)
    (Padded timesteps are 100% skipped by the LSTM kernel — zero hidden-state corruption)
   │
   ▼
[3] Current-Flow Anchoring & Masked Temporal Attention:
    - Extract current flow hidden state h_t = H[b, lengths[b] - 1, :]  ∈ R^64
    - Scaled dot-product attention: Query = W_q(h_t), Key = W_k(H)
    - Padded positions masked to -inf → softmax weights strictly 0.0 on padding
    - Context vector c_t = Σ α_i H_i  ∈ R^64
   │
   ▼
[4] Classification Head:
    Concat([h_t, c_t]) ∈ R^128 → Linear(128 → 64) → LayerNorm(64) → ReLU → Dropout(0.3) → Linear(64 → 4)
   │
   ▼
Output Logits: (B, 4) over ['bruteforce', 'normal', 'scan', 'synflood']
```

**Why current-step anchoring matters:** Concatenating `h_t` (exact current flow) with `c_t` (attention-weighted causal context) ensures the model never blurs sudden class transitions (e.g., the first flow of a brute-force or scan burst immediately following normal traffic) while still leveraging multi-flow burst context.

**Verified Padding Invariance:** Unit tests in [tests/test_model.py](tests/test_model.py) prove mathematically and empirically that corrupting padded slots with `±9999.0` or truncating the tensor to its unpadded length yields identical logits and attention weights (`atol=1e-5`), with strictly `0.0` attention weight on all padded steps.

---

## 7. Dataset Description

Uses the exact shared Pipeline v2 dataset (`cybersentinel_code/v2/data/flows/all_flows.csv`, unmodified):

- **Shape:** `10,676 rows × 74 columns`, `0` missing/NaN values.
- **Columns:** `round` (1) + Envelope metadata (6) + Behavioral features (66) + `label` (1).
- **Split rule (strictly round-based, zero overlap):**
  - **Training:** Rounds `3, 4, 5` — **7,941 flows**
  - **Testing (held out):** Round `6` — **2,735 flows**

| Round | Role | `bruteforce` | `normal` | `scan` | `synflood` | Total |
|---|---|---:|---:|---:|---:|---:|
| 3 | Train | 34 | 127 | 1,002 | 1,247 | 2,410 |
| 4 | Train | 33 | 458 | 1,002 | 1,317 | 2,810 |
| 5 | Train | 33 | 434 | 1,002 | 1,252 | 2,721 |
| **3–5 Total** | **Train** | **100 (1.26%)** | **1,019 (12.83%)** | **3,006 (37.85%)** | **3,816 (48.05%)** | **7,941** |
| **6** | **Test (Held-Out)** | **33 (1.21%)** | **445 (16.27%)** | **1,002 (36.64%)** | **1,255 (45.89%)** | **2,735** |
| **All** | **Total** | **133** | **1,464** | **4,008** | **5,071** | **10,676** |

---

## 8. Feature Selection

- Uses **strictly all 66 Pipeline v2 behavioral features** imported directly from [features_v2.py](../../features_v2.py) (`58` NFStream statistical features + `3` engineered rates/ratios + `5` 10-second behavior-window counts).
- **Strictly excluded from model input tensors** (enforced by `validate_feature_columns()` and unit tests in [tests/test_dataset.py](tests/test_dataset.py)):
  - `src_ip`, `src_port`, `dst_ip`, `dst_port`, `bidirectional_first_seen_ms`, `bidirectional_last_seen_ms`
  - `round` (splitting/grouping only)
  - `label` (target only)

---

## 9. Sequence Construction

Implemented in [sequence_builder.py](sequence_builder.py):

- **Stream Grouping:** Flows are partitioned strictly by `(round, src_ip)`. `src_ip` is used **only** as the grouping key to separate independent host streams (exactly mirroring Pipeline v2's per-source `WindowTracker`) and **never enters the 66-feature tensor**.
- **Chronological Causal Ordering:** Because NFStream emits flows on expiration, raw CSV rows are not time-monotonic. Within each `(round, src_ip)` stream, flows are stably sorted by `(bidirectional_last_seen_ms, bidirectional_first_seen_ms)` — the exact completion/arrival order observed by a live IDS.
- **Causal Many-to-One Windows (`L = 8`, stride `= 1`):** For every flow $t$, the sequence consists of up to `8` flows `[x_{max(0, t-7)}, ..., x_t]` placed contiguously in slots `[0 .. eff_len - 1]`, predicting strictly the current flow's label $y_t$.
- **Zero Future / Cross-Round Leakage:** Sequences never peek at flow $t+1$ and never cross round or `src_ip` boundaries.
- **100% Flow Preservation:** Early/cold-start flows with `< 8` history steps are right-padded with zeros (`eff_len` passed to `pack_padded_sequence`), preserving all **2,735 Round 6 test flows** for an exact 1-to-1 comparison with XGBoost.
- **Streaming Buffer Parity:** `CausalFlowBuffer` implements live single-flow rolling history per `src_ip` and is verified in [evaluate.py](evaluate.py) to match batch causal predictions across all 2,735 Round 6 flows (`streaming_matches_batch = True`).

---

## 10. Preprocessing

Implemented in [preprocessing.py](preprocessing.py) (`FlowFeaturePreprocessor`), fitted **strictly on training data only** (inside each CV fold and on Rounds 3–5 for final training; Round 6 is never touched):

1. Extract only the 66 validated behavioral feature columns.
2. Replace any `NaN`/`±inf` values with `0.0`.
3. Apply smooth monotone compression $\text{signed\_log1p}(x) = \text{sign}(x)\ln(1 + |x|)$ to stabilize heavy-tailed byte/packet/rate features.
4. Standardize per feature via z-score `(x - mean) / std`, explicitly clamping the 9 constant TCP flag columns (`std < 1e-12`, e.g., CWR/ECE/URG counts always 0) to `0.0` so no `NaN`/`inf` is produced.

---

## 11. Class Imbalance Handling

**Why NOT SMOTE on sequences:** Interpolating individual rows destroys real temporal trajectories across `(round, src_ip)` streams, and interpolating whole sequences fabricates unrealistic multi-flow histories. Furthermore, applying SMOTE before splitting (as in Berhanu et al., 2025) introduces data leakage.

**Selected Method:** **Balanced Inverse-Frequency Class-Weighted Cross-Entropy Loss** computed strictly from training labels (`w_c = N / (K * N_c)`), preserving exact real temporal sequences while never touching or oversampling Round 6.

**Verified Training Split Class Distribution Before & After Loss Weighting (Rounds 3–5, `N = 7,941`):**

| Class | Raw Count | Raw Proportion | Inverse-Freq Loss Weight (`w_c`) | Effective Weighted Loss Share |
|---|---:|---:|---:|---:|
| `bruteforce` | 100 | 1.26% | **19.8525** | 25.00% |
| `normal` | 1,019 | 12.83% | 1.9482 | 25.00% |
| `scan` | 3,006 | 37.85% | 0.6604 | 25.00% |
| `synflood` | 3,816 | 48.05% | 0.5202 | 25.00% |

---

## 12. Model Training

Implemented in [train.py](train.py) and executed in the repository-local `.venv` on **`NVIDIA GeForce RTX 3050 Laptop GPU`** (`torch 2.14.1+cu126`, Python 3.13.13):

- **Reproducibility:** Fixed deterministic seed `42` across Python, NumPy, PyTorch, and cuDNN.
- **Optimization:** `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`), `CosineAnnealingLR`, gradient clipping `1.0`, `batch_size = 64`, `max_epochs = 40`, early stopping `patience = 8` on validation Macro-F1 (with validation NLL tie-breaker).
- **Stage 1 — 3-Fold Leave-One-Round-Out CV (Rounds 3, 4, 5 only; zero Round 6 contact):**
  - Preprocessor and class weights are refit from scratch inside each fold.

| CV Fold (Validation Round) | Train Rounds | Best Epoch | Val Accuracy | Val Macro-F1 | `bruteforce` Recall | Normal FAR | Val NLL |
|---|---|---:|---:|---:|---:|---:|---:|
| Round 3 | 4, 5 | 18 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000042 |
| Round 4 | 3, 5 | 30 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000017 |
| Round 5 | 3, 4 | 26 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.000025 |
| **CV Mean** | — | **Selected: 25** | **1.0000** | **1.0000** | **1.0000** | **0.0000** | — |

- **Stage 2 — Final Model Fit (All Training Rounds 3, 4, 5):**
  - Trained for CV-selected `25` epochs in **`36.44 s`** on RTX 3050 (final train loss `0.000370`).
  - Saved separately to `artifacts/g2_lstm_attention.pt` (**`354.23 KB`**, `87,812` parameters), leaving `v2/models/best_xgb.json` completely untouched.

---

## 13. Evaluation Methodology

Implemented in [evaluate.py](evaluate.py) and [metrics.py](metrics.py):

- **Three Models Evaluated on Identical Held-Out Round 6 (`2,735` flows):**
  1. **XGBoost Baseline (22 features, shipped):** Loaded read-only from `cybersentinel_code/v2/models/best_xgb.json`.
  2. **XGBoost Baseline (66 features):** Separately trained on Rounds 3–5 using all 66 features and identical `XGBClassifier` hyperparameters (`n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist", n_jobs=1`), saved separately in `artifacts/xgb_66_reference.json`.
  3. **G2 Causal LSTM-Attention (66 features, `L = 8`):** Loaded from `artifacts/g2_lstm_attention.pt`.
- **Normal False-Alarm Rate (FAR):** `FP_normal / N_normal` = normal flows misclassified as any attack divided by total actual normal flows (`445` in Round 6).
- **Actual Single-Flow Inference Latency Measurement:** Measured strictly separate from training via `time.perf_counter()` across `500` individual Round 6 flows (after `20` warm-up flows, never batched):
  - **XGBoost (22 & 66 feat):** Single `(1, F)` NumPy row call to `predict()`.
  - **G2 Causal LSTM-Attention (CPU & GPU):** Full end-to-end single-flow causal streaming pipeline per arriving flow: `FlowFeaturePreprocessor.transform_single()` → `CausalFlowBuffer.push_and_build(src_ip, scaled_vec)` → `torch.from_numpy()` → `CausalLSTMAttention.forward()` → `argmax`, plus a model-only breakdown.
  - **Context Requirement:** G2 maintains an in-memory rolling deque of up to `L - 1 = 7` preceding completed flows per `src_ip` (handling cold-start hosts from flow #1 via exact length packing).

---

## 14. Actual Experimental Results (Verified — Held-Out Round 6)

All numbers below are **verified measured outputs** recorded in [artifacts/evaluation_results.json](artifacts/evaluation_results.json) and [artifacts/training_report.json](artifacts/training_report.json):

- **G2 Causal LSTM-Attention (66 features, `L = 8`) on Round 6 (`2,735` flows):**
  - **Accuracy:** `1.0000` (`2,735 / 2,735`)
  - **Macro-F1:** `1.0000`
  - **Per-Class Recall:** `bruteforce = 1.0000` (`33/33`), `normal = 1.0000` (`445/445`), `scan = 1.0000` (`1,002/1,002`), `synflood = 1.0000` (`1,255/1,255`)
  - **Normal False-Alarm Rate:** `0.0000` (`0 / 445` false positives)
  - **Streaming-vs-Batch Exact Consistency:** `True`
  - **CPU Single-Flow Latency (End-to-End with Preprocessing + Causal History Buffer):** `1.2773 ms` mean (`1.1857 ms` median, `1.8880 ms` p95)
  - **CPU Single-Flow Latency (Model Forward Only):** `1.1570 ms` mean (`0.120 ms` preprocessing + buffer overhead)
  - **GPU (RTX 3050) Single-Flow Latency (End-to-End, `batch_size = 1`):** `3.2386 ms` mean (`2.7341 ms` model-only; slower than CPU at `batch=1` due to host-to-device transfer and CUDA kernel launch overhead)
  - **Model Size:** `354.2 KB` (`87,812` parameters)

- **G2 Classification Report (Round 6):**
```text
              precision    recall  f1-score   support

  bruteforce     1.0000    1.0000    1.0000        33
      normal     1.0000    1.0000    1.0000       445
        scan     1.0000    1.0000    1.0000      1002
    synflood     1.0000    1.0000    1.0000      1255

    accuracy                         1.0000      2735
   macro avg     1.0000    1.0000    1.0000      2735
weighted avg     1.0000    1.0000    1.0000      2735
```

---

## 15. XGBoost Comparison (Held-Out Round 6, 2,735 Flows)

| Metric | XGBoost Baseline (22 feat, shipped) | XGBoost Baseline (66 feat) | G2 Causal LSTM-Attention (66 feat, L=8) |
|---|---:|---:|---:|
| **Accuracy** | 1.0000 | 1.0000 | 1.0000 |
| **Macro-F1** | 1.0000 | 1.0000 | 1.0000 |
| **`normal` Recall** | 1.0000 | 1.0000 | 1.0000 |
| **`synflood` Recall** | 1.0000 | 1.0000 | 1.0000 |
| **`scan` Recall** | 1.0000 | 1.0000 | 1.0000 |
| **`bruteforce` Recall** | 1.0000 | 1.0000 | 1.0000 |
| **False Alarm Rate (`normal`)** | 0.0000 | 0.0000 | 0.0000 |
| **Prediction Time — CPU (ms/flow)** | 0.6435 | 0.5840 | 1.2773 *(e2e w/ history)* |
| **Model Size (KB)** | 477.8 | 463.9 | 354.2 |

**Comparative Analysis:**
- **Detection accuracy parity:** All three models achieve `1.0000` Macro-F1 and `0.0000` normal FAR on held-out Round 6, including `100%` recall on all `33` minority `bruteforce` flows.
- **Latency trade-off:** Both XGBoost baselines remain ~`2.0–2.2×` faster per flow on CPU (`0.584–0.644 ms` vs `1.277 ms` end-to-end for G2), confirming the literature expectation (`GROUPING_ANALYSIS.md`) that gradient-boosted trees hold a speed advantage on single tabular flows. However, G2's `1.28 ms/flow` CPU latency still sits comfortably inside CyberSentinel v2's `5 ms/flow` live budget.
- **Footprint & temporal modeling:** G2's checkpoint (`354.2 KB`) is **25.9% smaller** than the shipped XGBoost JSON (`477.8 KB`), natively captures multi-flow temporal evolution across up to 8 flows per host, and exposes per-timestep temporal attention weights.

---

## 16. Confusion Matrix

**Verified Round 6 Confusion Matrix — Identical for G2 Causal LSTM-Attention, XGBoost (66 feat), and Shipped XGBoost (22 feat):**

| True \ Predicted | `bruteforce` | `normal` | `scan` | `synflood` | Total |
|---|---:|---:|---:|---:|---:|
| **`bruteforce`** | **33** | 0 | 0 | 0 | 33 |
| **`normal`** | 0 | **445** | 0 | 0 | 445 |
| **`scan`** | 0 | 0 | **1,002** | 0 | 1,002 |
| **`synflood`** | 0 | 0 | 0 | **1,255** | 1,255 |
| **Total** | 33 | 445 | 1,002 | 1,255 | **2,735** |

Side-by-side confusion matrix plot saved at [artifacts/confusion_matrices.png](artifacts/confusion_matrices.png).

---

## 17. Limitations

1. **Lab-environment ceiling (honest caveat):** Consistent with `README.md` Section 16 and `PIPELINE_V2.md`, all four recorded rounds used fixed attack parameters (`nmap`, `hydra`, `hping3`). Achieving `1.0000` Macro-F1 on Round 6 proves pipeline correctness and generalization across separate lab captures, **not** 100% real-world accuracy on noisy enterprise traffic or unseen attack variants.
2. **Small minority class sample size:** Only `100` training and `33` test flows exist for `bruteforce`; confidence intervals on minority-class performance would narrow significantly with more varied brute-force captures.
3. **Stateful history requirement:** Unlike stateless per-flow XGBoost, G2 maintains an in-memory per-`src_ip` deque (`maxlen = 8`). Under source-IP spoofing (`hping3 --rand-source`), every spoofed IP starts at `length = 1` (falling back to single-flow classification without multi-flow context).
4. **Latency overhead vs. XGBoost:** G2 takes `1.28 ms/flow` on CPU (`2.2×` slower than 66-feature XGBoost's `0.58 ms/flow`), and single-flow GPU inference is slower than CPU (`3.24 ms/flow`) due to `batch_size = 1` CUDA launch/transfer overhead.

---

## 18. Future Improvements

1. **Harder, parameter-varied test rounds & zero-day leave-one-attack-out testing:** Evaluate G2 vs. XGBoost when scan rates, brute-force concurrency, or attack types differ substantially between training and test sessions (where temporal sequence dynamics should differentiate models more sharply).
2. **True $O(1)$ incremental LSTM hidden-state streaming:** Persist per-`src_ip` `(h, c)` states across arriving flows instead of re-evaluating the 8-step window per flow to cut CPU latency further.
3. **Lightweight Small Transformer ablation:** Compare `CausalLSTMAttention` against a 1–2 layer causal Transformer with Feature Positional Encoding (Zaim & Yolaçan, 2025) on harder multi-session captures.
4. **Attention + SHAP interpretability (G5 integration):** Visualize temporal attention weights alongside feature attributions to explain *which* preceding flows and features triggered an alert.

---

## 19. Installation Instructions

Run inside the repository root (`Network_Intrusion_AISD`) using an isolated repository-local virtual environment (does not modify global Python):

```powershell
# 1. Create and activate repository-local virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install PyTorch (CUDA 12.6 for NVIDIA RTX GPUs, or drop --extra-index-url for CPU)
python -m pip install torch==2.14.1+cu126 numpy pandas scikit-learn xgboost matplotlib pytest --extra-index-url https://download.pytorch.org/whl/cu126
```

---

## 20. Commands to Reproduce the Experiment

All commands run from the `Network_Intrusion_AISD` repository root using `.venv`:

```powershell
# 1. Run G2 unit tests (17 tests: dataset, features, zero leakage, sequences, padding invariance, model, metrics)
.\.venv\Scripts\python.exe -m pytest cybersentinel_code/v2/experiments/g2_sequence_attention/tests -v

# 2. Run existing Pipeline v2 tests (40 tests: verify zero regressions)
.\.venv\Scripts\python.exe -m pytest cybersentinel_code/v2/tests -q

# 3. Train G2 model (3-fold Leave-One-Round-Out CV on Rounds 3-5 + final fit -> artifacts/)
.\.venv\Scripts\python.exe cybersentinel_code/v2/experiments/g2_sequence_attention/train.py

# 4. Run fair Round 6 evaluation against Shipped XGBoost (22 feat) & XGBoost (66 feat)
.\.venv\Scripts\python.exe cybersentinel_code/v2/experiments/g2_sequence_attention/evaluate.py
```

### Verified Test Suite Summary
- **G2 Tests (`tests/`):** `17/17 passed` (`0` warnings)
- **Existing Pipeline v2 Tests (`v2/tests/`):** `40/40 passed` (`0` regressions)
