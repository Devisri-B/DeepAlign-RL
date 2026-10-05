"""Composite Multi-Objective Verifiable Reward Aggregator."""

from typing import Dict, List, Optional
from deepalign_rl.rewards.base import BaseRewardVerifier, RewardResult


class CompositeReward(BaseRewardVerifier):
    """Combines multiple reward verifiers into a weighted composite score."""

    def __init__(self, verifiers: List[BaseRewardVerifier], name: str = "composite_reward"):
        super().__init__(name=name, weight=1.0)
        self.verifiers = verifiers

    def evaluate(self, prompt: str, completion: str, ground_truth: Optional[str] = None) -> RewardResult:
        sub_results: Dict[str, RewardResult] = {}
        total_score = 0.0
        all_correct = True

        for verifier in self.verifiers:
            res = verifier.evaluate(prompt, completion, ground_truth)
            sub_results[verifier.name] = res
            total_score += res.score
            if res.is_correct is False:
                all_correct = False

        breakdown = {name: res.score for name, res in sub_results.items()}
        details = {
            "breakdown": breakdown,
            "components": {name: res.details for name, res in sub_results.items()},
        }

        return RewardResult(
            score=total_score,
            is_correct=all_correct,
            details=details,
            verifier_name=self.name,
        )
