"""TDD tests for CausalLSTMAttention architecture, strict padding invariance, and training loop."""
from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pytest
import torch

G2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_DIR = os.path.dirname(os.path.dirname(G2_DIR))
for _p in (V2_DIR, G2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as g2_cfg
import model as g2_model
import train as g2_train


def test_model_input_output_dimensions_and_attention_normalization():
    g2_train.set_seed(42)
    net = g2_model.CausalLSTMAttention(
        input_dim=66,
        num_classes=4,
        embed_dim=64,
        hidden_dim=64,
        num_layers=2,
    )
    net.eval()

    x = torch.randn(5, 8, 66)
    lengths = torch.tensor([1, 3, 5, 8, 2], dtype=torch.long)
    with torch.no_grad():
        logits, attn = net(x, lengths, return_attention=True)

    assert logits.shape == (5, 4)
    assert attn.shape == (5, 8)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attn).all()

    # Attention weights must sum to 1 over valid steps and be strictly 0 on padded steps
    for i, L_eff in enumerate(lengths.tolist()):
        assert pytest.approx(float(attn[i, :L_eff].sum()), abs=1e-5) == 1.0
        if L_eff < 8:
            assert float(attn[i, L_eff:].abs().max()) == 0.0


def test_padding_timesteps_do_not_influence_lstm_hidden_states_or_logits():
    """Requirement 1: Zero-padded (or arbitrarily corrupted) padded timesteps must have ZERO effect."""
    g2_train.set_seed(123)
    net = g2_model.CausalLSTMAttention(input_dim=66, num_classes=4)
    net.eval()

    real_3 = torch.randn(2, 3, 66)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    # Case A: Clean zero-padded to L=8
    x_zero_pad = torch.zeros(2, 8, 66)
    x_zero_pad[:, :3, :] = real_3
    # Zero out step 2 for sample 1 (since its valid length is 2)
    x_zero_pad[1, 2, :] = 0.0

    # Case B: Garbage values in padded positions (must NOT change hidden state or logits!)
    x_garbage_pad = x_zero_pad.clone()
    x_garbage_pad[0, 3:, :] = 9999.0
    x_garbage_pad[1, 2:, :] = -5432.0

    # Case C: Unpadded tensor with max length = 3
    x_short = real_3.clone()
    x_short[1, 2, :] = 0.0

    with torch.no_grad():
        logits_zero, attn_zero = net(x_zero_pad, lengths, return_attention=True)
        logits_garbage, attn_garbage = net(x_garbage_pad, lengths, return_attention=True)
        logits_short, attn_short = net(x_short, lengths, return_attention=True)

    torch.testing.assert_close(logits_zero, logits_garbage, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(attn_zero, attn_garbage, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits_zero, logits_short, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(attn_zero[:, :3], attn_short, atol=1e-5, rtol=1e-5)


def test_single_flow_cold_start_length_one():
    g2_train.set_seed(7)
    net = g2_model.CausalLSTMAttention(input_dim=66, num_classes=4)
    net.eval()

    x = torch.zeros(1, 8, 66)
    x[0, 0, :] = torch.randn(66)
    lengths = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        logits, attn = net(x, lengths, return_attention=True)

    assert logits.shape == (1, 4)
    assert pytest.approx(float(attn[0, 0]), abs=1e-6) == 1.0
    assert float(attn[0, 1:].abs().max()) == 0.0


def test_deterministic_seeding_and_one_epoch_training_and_checkpoint_roundtrip():
    cfg = g2_cfg.G2Config(batch_size=16, max_epochs=2)
    g2_train.set_seed(cfg.seed)
    m1 = g2_model.CausalLSTMAttention(input_dim=66, num_classes=4)
    g2_train.set_seed(cfg.seed)
    m2 = g2_model.CausalLSTMAttention(input_dim=66, num_classes=4)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1, p2)

    # Synthetic separable 4-class sequence batch
    rng = np.random.default_rng(42)
    n = 64
    y = np.tile(np.arange(4, dtype=np.int64), n // 4)
    X = np.zeros((n, 8, 66), dtype=np.float32)
    lengths = np.full(n, 4, dtype=np.int64)
    for i in range(n):
        X[i, :4, :] = rng.normal(loc=float(y[i]) * 2.0, scale=0.2, size=(4, 66)).astype(np.float32)

    loader = g2_train.make_dataloader(X, lengths, y, batch_size=16, shuffle=True, seed=42)
    weights = torch.ones(4, dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(m1.parameters(), lr=5e-3)
    device = torch.device("cpu")

    loss1 = g2_train.train_one_epoch(m1, loader, criterion, optimizer, device, grad_clip=1.0)
    loss2 = g2_train.train_one_epoch(m1, loader, criterion, optimizer, device, grad_clip=1.0)
    assert np.isfinite(loss1) and np.isfinite(loss2)
    assert loss2 < loss1

    # Save & reload checkpoint and verify identical predictions
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "test_ckpt.pt")
        torch.save(m1.state_dict(), ckpt_path)
        m_loaded = g2_model.CausalLSTMAttention(input_dim=66, num_classes=4)
        m_loaded.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        m1.eval()
        m_loaded.eval()
        xb = torch.from_numpy(X[:8])
        lb = torch.from_numpy(lengths[:8])
        with torch.no_grad():
            torch.testing.assert_close(m1(xb, lb), m_loaded(xb, lb))
