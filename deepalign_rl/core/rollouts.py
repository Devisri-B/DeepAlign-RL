"""Rollout collection and generation pipeline for Group Relative Policy Optimization."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from deepalign_rl.core.grpo import compute_group_advantages
from deepalign_rl.models.policy import PolicyModelWrapper
from deepalign_rl.rewards.base import BaseRewardVerifier, RewardResult


@dataclass
class RolloutBatch:
    """Container holding a batch of generated rollouts and computed group statistics."""
    prompts: List[str]
    completions: List[str]
    ground_truths: List[Optional[str]]
    input_ids: torch.Tensor          # [B*G, S]
    attention_mask: torch.Tensor     # [B*G, S]
    response_mask: torch.Tensor      # [B*G, S-1] (1 for generated tokens, 0 for prompt/pad)
    old_policy_logps: torch.Tensor   # [B*G, S-1]
    ref_logps: torch.Tensor          # [B*G, S-1]
    raw_rewards: torch.Tensor        # [B*G]
    advantages: torch.Tensor         # [B*G]
    reward_results: List[RewardResult]


class RolloutEngine:
    """Manages parallel group generation and verifiable evaluation."""

    def __init__(
        self,
        model_wrapper: PolicyModelWrapper,
        reward_verifier: BaseRewardVerifier,
        group_size: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.model = model_wrapper
        self.reward_verifier = reward_verifier
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def collect_rollouts(
        self,
        prompts: List[str],
        ground_truths: Optional[List[Optional[str]]] = None,
    ) -> RolloutBatch:
        """Sample group completions, evaluate verifiable rewards, and compute advantages."""
        if ground_truths is None:
            ground_truths = [None] * len(prompts)

        # 1. Sample G generations per prompt
        completions, input_ids, attention_mask, prompt_lengths = self.model.generate_responses(
            prompt_texts=prompts,
            num_generations=self.group_size,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        total_samples = len(completions)
        seq_len = input_ids.size(1)

        # 2. Build response mask for loss calculation
        # Targets are shifted by 1: input_ids[:, 1:], so seq_len - 1
        response_mask = torch.zeros(
            (total_samples, seq_len - 1),
            dtype=torch.float32,
            device=self.model.device,
        )

        for i in range(total_samples):
            p_len = prompt_lengths[i]
            # Generated tokens begin at p_len - 1 in the shifted target space
            # and end where attention_mask becomes 0
            valid_len = int(attention_mask[i].sum().item())
            start_idx = max(0, p_len - 1)
            end_idx = max(0, valid_len - 1)
            if end_idx > start_idx:
                response_mask[i, start_idx:end_idx] = 1.0

        # 3. Compute log-probabilities under current policy and reference policy
        with torch.no_grad():
            old_policy_logps = self.model.get_token_log_probs(
                self.model.policy_model, input_ids, attention_mask
            )
            ref_logps = self.model.get_ref_token_log_probs(
                input_ids, attention_mask
            )

        # 4. Evaluate verifiable rewards
        # Expand prompts and ground truths to match completions (B * G)
        expanded_prompts = []
        expanded_gts = []
        for p, gt in zip(prompts, ground_truths):
            for _ in range(self.group_size):
                expanded_prompts.append(p)
                expanded_gts.append(gt)

        reward_results = [
            self.reward_verifier.evaluate(p, c, gt)
            for p, c, gt in zip(expanded_prompts, completions, expanded_gts)
        ]
        raw_rewards = torch.tensor(
            [res.score for res in reward_results],
            dtype=torch.float32,
            device=self.model.device,
        )

        # 5. Compute group standardized advantages
        advantages = compute_group_advantages(
            raw_rewards, group_size=self.group_size
        )

        return RolloutBatch(
            prompts=expanded_prompts,
            completions=completions,
            ground_truths=expanded_gts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            old_policy_logps=old_policy_logps,
            ref_logps=ref_logps,
            raw_rewards=raw_rewards,
            advantages=advantages,
            reward_results=reward_results,
        )
