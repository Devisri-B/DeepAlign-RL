"""Base classes and interfaces for verifiable rewards."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RewardResult(BaseModel):
    """Result of evaluating a single completion."""

    score: float = Field(description="Numerical reward value (typically in [0, 1] or normalized)")
    is_correct: Optional[bool] = Field(default=None, description="Whether the response is strictly correct")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic breakdown for inspection")
    verifier_name: str = Field(default="base", description="Name of the verifier that produced this result")


class BaseRewardVerifier(ABC):
    """Abstract base class for verifiable rule-based reward functions."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, prompt: str, completion: str, ground_truth: Optional[str] = None) -> RewardResult:
        """Evaluate a single completion given the prompt and optional ground truth.

        Args:
            prompt: Input query or prompt.
            completion: Generated response from policy.
            ground_truth: Target answer or test cases.

        Returns:
            RewardResult containing score and diagnostics.
        """
        pass

    def batch_evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[Optional[str]]] = None,
    ) -> List[RewardResult]:
        """Batch evaluation convenience method."""
        if ground_truths is None:
            ground_truths = [None] * len(prompts)
        return [
            self.evaluate(p, c, gt)
            for p, c, gt in zip(prompts, completions, ground_truths)
        ]
