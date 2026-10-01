"""DeepAlign-RL: Reasoning & Alignment Post-Training Engine from Scratch.

A modular, research-grade post-training engine implementing:
- DeepSeek-style Group Relative Policy Optimization (GRPO)
- Direct Preference Optimization (DPO)
- Verifiable Rule-Based Rewards (SymPy Math, AST Logic, Format/CoT Enforcers)
"""

__version__ = "0.1.0"
__author__ = "DeepAlign-RL Team"

from deepalign_rl.core.grpo import GRPOTrainer, GRPOLoss
from deepalign_rl.core.dpo import DPOTrainer, DPOLoss
from deepalign_rl.rewards.math_verifier import VerifiableMathReward
from deepalign_rl.rewards.format_verifier import FormatReward
from deepalign_rl.rewards.composite import CompositeReward

__all__ = [
    "GRPOTrainer",
    "GRPOLoss",
    "DPOTrainer",
    "DPOLoss",
    "VerifiableMathReward",
    "FormatReward",
    "CompositeReward",
]
