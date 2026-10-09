"""Unit tests for DPO loss function."""

import torch
import pytest
from deepalign_rl.core.dpo import DPOLoss
from deepalign_rl.utils.config import DPOConfig


def test_dpo_loss_computation():
    config = DPOConfig(beta=0.1)
    loss_fn = DPOLoss(config=config)

    # Policy assigns higher probability to chosen
    policy_chosen = torch.tensor([-5.0, -4.0], requires_grad=True)
    policy_rejected = torch.tensor([-15.0, -12.0], requires_grad=True)
    ref_chosen = torch.tensor([-6.0, -5.0])
    ref_rejected = torch.tensor([-7.0, -6.0])

    loss, metrics = loss_fn(
        policy_chosen_logps=policy_chosen,
        policy_rejected_logps=policy_rejected,
        ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected,
    )

    assert not torch.isnan(loss)
    loss.backward()
    assert policy_chosen.grad is not None
    assert metrics["accuracy"] == 1.0
    assert metrics["reward_margin"] > 0
