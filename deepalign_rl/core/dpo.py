"""Direct Preference Optimization (DPO) implementation from scratch.

Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepalign_rl.utils.config import DPOConfig


class DPOLoss(nn.Module):
    """Computes the Direct Preference Optimization (DPO) loss."""

    def __init__(self, config: Optional[DPOConfig] = None):
        super().__init__()
        self.config = config or DPOConfig()

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss and tracking metrics.

        Args:
            policy_chosen_logps: Sum of log-probs for chosen completions under current policy [B].
            policy_rejected_logps: Sum of log-probs for rejected completions under current policy [B].
            ref_chosen_logps: Sum of log-probs for chosen completions under reference policy [B].
            ref_rejected_logps: Sum of log-probs for rejected completions under reference policy [B].

        Returns:
            loss: Scalar DPO loss.
            metrics: Diagnostic metrics (rewards, margins, accuracy).
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        # DPO scaled logits
        scaled_logits = self.config.beta * logits

        # Bradley-Terry loss: -log sigma(beta * (pi_ratio - ref_ratio))
        if self.config.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(scaled_logits) * (1 - self.config.label_smoothing)
                - F.logsigmoid(-scaled_logits) * self.config.label_smoothing
            )
        elif self.config.loss_type == "hinge":
            losses = torch.relu(1 - scaled_logits)
        elif self.config.loss_type == "ipo":
            # IPO loss: (scaled_logits - 1 / (2 * beta)) ** 2
            losses = (logits - 1 / (2 * self.config.beta)) ** 2
        else:
            losses = -F.logsigmoid(scaled_logits)

        loss = losses.mean()

        with torch.no_grad():
            chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
            reward_margin = chosen_rewards - rejected_rewards
            accuracy = (reward_margin > 0).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": reward_margin.mean().item(),
            "accuracy": accuracy,
        }

        return loss, metrics


# Export for convenience
DPOTrainer = None  # Defined in trainers/dpo_trainer.py
