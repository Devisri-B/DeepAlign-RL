"""Unit tests for GRPO advantage computation and loss."""

import torch
import pytest
from deepalign_rl.core.grpo import compute_group_advantages, GRPOLoss
from deepalign_rl.utils.config import GRPOConfig


def test_compute_group_advantages():
    # 2 groups of size 4
    # Group 1: rewards [1.0, 1.0, 0.0, 0.0] -> mean = 0.5, std = 0.5
    # Group 2: rewards [0.0, 0.0, 0.0, 0.0] -> zero variance, advantages should be 0
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    adv = compute_group_advantages(rewards, group_size=4)

    assert adv.shape == rewards.shape
    # Check group 1 has mean 0
    assert torch.isclose(adv[:4].mean(), torch.tensor(0.0), atol=1e-5)
    # Check group 1 positive values > 0, negative values < 0
    assert adv[0] > 0
    assert adv[2] < 0

    # Check zero variance group has all 0 advantages
    assert torch.all(adv[4:] == 0.0)

    # Size mismatch test (e.g. 4 samples with group_size=5)
    odd_rewards = torch.tensor([1.5, 1.0, 0.5, 0.0], dtype=torch.float32)
    adv_mismatch = compute_group_advantages(odd_rewards, group_size=5)
    assert adv_mismatch.shape == odd_rewards.shape
    assert adv_mismatch[0] > 0
    assert adv_mismatch[-1] < 0


def test_grpo_loss_backward():
    config = GRPOConfig(clip_eps=0.2, beta_kl=0.04)
    loss_fn = GRPOLoss(config=config)

    batch_size = 4  # e.g. 1 group of 4
    seq_len = 10

    # Policy log-probs with requires_grad
    policy_logps = torch.randn((batch_size, seq_len), requires_grad=True)
    old_policy_logps = policy_logps.detach().clone()
    ref_logps = policy_logps.detach().clone()
    advantages = torch.tensor([1.0, 0.5, -0.5, -1.0], dtype=torch.float32)
    response_mask = torch.ones((batch_size, seq_len), dtype=torch.float32)

    loss, metrics = loss_fn(
        policy_logps=policy_logps,
        old_policy_logps=old_policy_logps,
        ref_logps=ref_logps,
        advantages=advantages,
        response_mask=response_mask,
    )

    assert not torch.isnan(loss)
    loss.backward()
    assert policy_logps.grad is not None
    assert not torch.isnan(policy_logps.grad).any()
    assert "policy_loss" in metrics
    assert "kl_divergence" in metrics
