"""Unit tests for RolloutEngine and DummyCausalLM."""

import torch
import pytest
from deepalign_rl.core.rollouts import RolloutEngine
from deepalign_rl.models.policy import PolicyModelWrapper
from deepalign_rl.rewards.composite import CompositeReward
from deepalign_rl.rewards.format_verifier import FormatReward
from deepalign_rl.rewards.math_verifier import VerifiableMathReward
from deepalign_rl.utils.config import ModelConfig


def test_rollout_engine_with_dummy_model():
    model_cfg = ModelConfig(is_dummy=True)
    model_wrapper = PolicyModelWrapper(model_cfg, device=torch.device("cpu"))

    reward_verifier = CompositeReward([
        VerifiableMathReward(weight=1.0),
        FormatReward(weight=0.5),
    ])

    engine = RolloutEngine(
        model_wrapper=model_wrapper,
        reward_verifier=reward_verifier,
        group_size=2,
        max_new_tokens=16,
    )

    prompts = ["Solve 2+2:", "Solve 3*3:"]
    ground_truths = ["4", "9"]

    batch = engine.collect_rollouts(prompts=prompts, ground_truths=ground_truths)

    # Total samples should be 2 prompts * 2 generations = 4
    assert len(batch.completions) == 4
    assert batch.input_ids.size(0) == 4
    assert batch.response_mask.size(0) == 4
    assert batch.advantages.size(0) == 4
    assert len(batch.reward_results) == 4
