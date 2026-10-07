"""DeepSeek-style Group Relative Policy Optimization (GRPO) implementation.

Key Features:
1. Critic-Free: Eliminates the value model by standardizing rewards across a group of G sampled rollouts.
2. Token-level Clipped Surrogate Objective: Stabilizes policy ratio divergence.
3. Schulman KL Divergence Penalty: Accurately constrains drift from the frozen reference policy.
4. Response Masking: Ensures optimization gradient flows strictly through generated tokens, not prompt tokens.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepalign_rl.utils.config import GRPOConfig


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute group-relative standardized advantages.

    Args:
        rewards: 1D or 2D tensor of raw scalar rewards of shape [N] where N = B * G.
        group_size: Number of generations G per prompt.
        eps: Small epsilon to prevent division by zero when variance is near zero.

    Returns:
        advantages: Tensor of same shape as rewards with group-mean 0 and std 1.
    """
    total_samples = rewards.size(0)
    num_groups = total_samples // group_size
    rewards_grouped = rewards.view(num_groups, group_size)

    # Compute mean and standard deviation per group
    mean = rewards_grouped.mean(dim=-1, keepdim=True)
    std = rewards_grouped.std(dim=-1, keepdim=True, unbiased=False)

    # Standardize advantages
    advantages = (rewards_grouped - mean) / (std + eps)

    # If std is 0 (all outputs in group got identical reward), advantages will be ~0
    zero_var_mask = (std < eps)
    advantages[zero_var_mask.expand_as(advantages)] = 0.0

    return advantages.view(total_samples)


class GRPOLoss(nn.Module):
    """Computes the Group Relative Policy Optimization (GRPO) loss."""

    def __init__(self, config: Optional[GRPOConfig] = None):
        super().__init__()
        self.config = config or GRPOConfig()

    def forward(
        self,
        policy_logps: torch.Tensor,
        old_policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO clipped loss and KL divergence penalty.

        Args:
            policy_logps: Tensor [B*G, S-1] of token log-probabilities under current policy.
            old_policy_logps: Tensor [B*G, S-1] under rollout policy (detached).
            ref_logps: Tensor [B*G, S-1] under frozen reference policy.
            advantages: Tensor [B*G] of group-normalized advantage values.
            response_mask: Tensor [B*G, S-1] (1 for response token, 0 for prompt or padding).

        Returns:
            loss: Scalar objective to be minimized.
            metrics: Diagnostic metrics (policy_loss, kl_loss, clip_frac, approx_kl).
        """
        # Shape alignment: broadcast advantages across token sequence dimension
        # advantages: [B*G, 1]
        adv = advantages.unsqueeze(-1)

        # Importance sampling ratio r_t(theta) = exp(log pi_theta - log pi_old)
        # Clamped for numerical stability
        log_ratio = policy_logps - old_policy_logps
        ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))

        # Clipped surrogate objective
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv
        surrogate_loss = torch.min(surr1, surr2)

        # Token-level KL divergence penalty
        if self.config.kl_penalty_type == "schulman":
            # Schulman unbiased estimator: exp(log_ref - log_pi) - (log_ref - log_pi) - 1
            log_ref_ratio = ref_logps - policy_logps
            ref_ratio = torch.exp(torch.clamp(log_ref_ratio, -10.0, 10.0))
            kl_divergence = ref_ratio - log_ref_ratio - 1.0
        else:
            # Reverse KL: log_pi - log_ref
            kl_divergence = policy_logps - ref_logps

        # Combine: maximizing (surrogate - beta * KL) <=> minimizing -(surrogate - beta * KL)
        per_token_objective = surrogate_loss - self.config.beta_kl * kl_divergence

        # Masking and normalization per completion
        masked_objective = per_token_objective * response_mask
        tokens_per_response = response_mask.sum(dim=-1).clamp(min=1.0)

        # Average per sequence, then average across the batch
        per_response_loss = - (masked_objective.sum(dim=-1) / tokens_per_response)
        total_loss = per_response_loss.mean()

        # Compute diagnostics
        with torch.no_grad():
            masked_surr = surrogate_loss * response_mask
            masked_kl = kl_divergence * response_mask
            total_tokens = response_mask.sum().clamp(min=1.0).item()

            policy_loss_val = - (masked_surr.sum().item() / total_tokens)
            kl_val = masked_kl.sum().item() / total_tokens
            clipped = (ratio < (1.0 - self.config.clip_eps)) | (ratio > (1.0 + self.config.clip_eps))
            clip_fraction = (clipped.float() * response_mask).sum().item() / total_tokens
            approx_kl = 0.5 * ((log_ratio ** 2) * response_mask).sum().item() / total_tokens

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss_val,
            "kl_divergence": kl_val,
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item() if advantages.numel() > 1 else 0.0,
        }

        return total_loss, metrics


# Export for convenience
GRPOTrainer = None  # Defined in trainers/grpo_trainer.py
