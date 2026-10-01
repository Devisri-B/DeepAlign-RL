"""Configuration definitions and YAML loaders for DeepAlign-RL."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for policy and reference models."""
    model_name_or_path: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", description="HF hub model or local path")
    torch_dtype: str = Field(default="bfloat16", description="Weights precision: float32, float16, bfloat16")
    device: Optional[str] = Field(default=None, description="Explicit device override: cuda, mps, cpu")
    use_flash_attention: bool = Field(default=False, description="Enable flash attention if supported")
    is_dummy: bool = Field(default=False, description="Use fast synthetic policy for testing and CI")


class GRPOConfig(BaseModel):
    """Hyperparameters for Group Relative Policy Optimization."""
    num_generations: int = Field(default=4, ge=2, description="Group size G: completions sampled per prompt")
    max_prompt_length: int = Field(default=512, description="Maximum token length for prompt")
    max_response_length: int = Field(default=512, description="Maximum token length for generated reasoning")
    temperature: float = Field(default=0.8, gt=0, description="Sampling temperature")
    top_p: float = Field(default=0.95, gt=0, le=1.0, description="Top-p nucleus sampling cutoff")
    learning_rate: float = Field(default=1e-6, gt=0, description="Policy learning rate")
    clip_eps: float = Field(default=0.2, gt=0, description="PPO clipping epsilon parameter")
    beta_kl: float = Field(default=0.04, ge=0, description="KL divergence penalty coefficient")
    kl_penalty_type: str = Field(default="schulman", description="kl estimator: 'schulman' or 'exact'")
    epochs_per_rollout: int = Field(default=1, ge=1, description="Optimization epochs per rollout batch")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Max gradient norm clipping")
    batch_size: int = Field(default=2, ge=1, description="Number of distinct prompts per optimization step")
    total_steps: int = Field(default=100, ge=1, description="Total training steps")
    save_every: int = Field(default=25, description="Checkpoint save frequency")
    eval_every: int = Field(default=10, description="Validation frequency")


class DPOConfig(BaseModel):
    """Hyperparameters for Direct Preference Optimization."""
    beta: float = Field(default=0.1, gt=0, description="DPO beta temperature parameter")
    learning_rate: float = Field(default=5e-7, gt=0, description="Policy learning rate")
    batch_size: int = Field(default=4, ge=1, description="Batch size for preference pairs")
    max_length: int = Field(default=1024, description="Max token length for prompt + response")
    max_prompt_length: int = Field(default=512, description="Max prompt token length")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Max gradient norm clipping")
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5, description="Label smoothing for preference labels")
    loss_type: str = Field(default="sigmoid", description="DPO loss variant: sigmoid, ipo, hinge")
    total_steps: int = Field(default=100, ge=1, description="Total training steps")
    eval_every: int = Field(default=10, description="Validation frequency")


class TrainingPipelineConfig(BaseModel):
    """Unified configuration container."""
    experiment_name: str = Field(default="deepalign_experiment")
    output_dir: str = Field(default="./outputs")
    model: ModelConfig = Field(default_factory=ModelConfig)
    grpo: Optional[GRPOConfig] = Field(default=None)
    dpo: Optional[DPOConfig] = Field(default=None)
    seed: int = Field(default=42)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingPipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
