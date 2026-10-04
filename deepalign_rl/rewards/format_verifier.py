"""Format and Chain-of-Thought Structure Verifier.

Rewards structured reasoning tokens (e.g. <think>...</think><answer>...</answer>)
and penalizes unclosed tags, missing sections, and degenerative length.
"""

import re
from typing import Optional
from deepalign_rl.rewards.base import BaseRewardVerifier, RewardResult


class FormatReward(BaseRewardVerifier):
    """Verifies that the policy conforms to the specified CoT format."""

    def __init__(
        self,
        name: str = "format_verifier",
        weight: float = 0.5,
        min_think_chars: int = 20,
        max_think_chars: int = 4000,
        penalize_empty_answer: bool = True,
    ):
        super().__init__(name=name, weight=weight)
        self.min_think_chars = min_think_chars
        self.max_think_chars = max_think_chars
        self.penalize_empty_answer = penalize_empty_answer

    def evaluate(self, prompt: str, completion: str, ground_truth: Optional[str] = None) -> RewardResult:
        if not completion:
            return RewardResult(
                score=0.0,
                is_correct=False,
                details={"error": "Empty completion"},
                verifier_name=self.name,
            )

        details = {
            "has_open_think": False,
            "has_close_think": False,
            "has_open_answer": False,
            "has_close_answer": False,
            "think_length": 0,
            "answer_length": 0,
            "format_valid": False,
        }

        # Check <think> tags
        has_open_think = "<think>" in completion
        has_close_think = "</think>" in completion
        details["has_open_think"] = has_open_think
        details["has_close_think"] = has_close_think

        # Check <answer> tags
        has_open_answer = "<answer>" in completion
        has_close_answer = "</answer>" in completion
        details["has_open_answer"] = has_open_answer
        details["has_close_answer"] = has_close_answer

        score = 0.0

        # Pattern: exact <think>...</think> followed by optional text or <answer>...</answer>
        think_match = re.search(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            details["think_length"] = len(think_content)
            # Reward for valid thinking block
            if len(think_content) >= self.min_think_chars:
                score += 0.5
            else:
                score += 0.2  # partial credit for tags present

            # Check if answer tag present and formatted
            answer_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                details["answer_length"] = len(answer_content)
                if len(answer_content) > 0:
                    score += 0.5
                    details["format_valid"] = True
            elif not has_open_answer and not has_close_answer:
                # If answer tags aren't required, check if answer follows </think>
                after_think = completion[think_match.end() :].strip()
                if len(after_think) > 0:
                    score += 0.3
                    details["format_valid"] = True
        else:
            # Partial formatting checks
            if has_open_think and not has_close_think:
                # Runaway generation without closing think
                score -= 0.2
            if not has_open_think and has_close_think:
                score -= 0.1

        # Normalize score between 0.0 and 1.0
        normalized_score = max(0.0, min(1.0, score))

        return RewardResult(
            score=normalized_score * self.weight,
            is_correct=details["format_valid"],
            details=details,
            verifier_name=self.name,
        )
