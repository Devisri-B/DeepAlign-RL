"""Full Training Loop and Engine for Group Relative Policy Optimization (GRPO)."""

import math
from pathlib import Path
from typing import List, Optional
import torch
from torch.optim import AdamW
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from deepalign_rl.core.grpo import GRPOLoss
from deepalign_rl.core.rollouts import RolloutEngine
from deepalign_rl.datasets.math_reasoning import MathExample
from deepalign_rl.models.policy import PolicyModelWrapper
from deepalign_rl.rewards.base import BaseRewardVerifier
from deepalign_rl.utils.config import GRPOConfig
from deepalign_rl.utils.logger import MetricLogger, console


class GRPOTrainer:
    """Orchestrates GRPO policy rollouts, verifiable scoring, and gradient updates."""

    def __init__(
        self,
        model_wrapper: PolicyModelWrapper,
        reward_verifier: BaseRewardVerifier,
        config: GRPOConfig,
        train_dataset: List[MathExample],
        val_dataset: Optional[List[MathExample]] = None,
        output_dir: str = "./outputs/grpo",
    ):
        self.model = model_wrapper
        self.reward_verifier = reward_verifier
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset or []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rollout_engine = RolloutEngine(
            model_wrapper=self.model,
            reward_verifier=self.reward_verifier,
            group_size=self.config.num_generations,
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        self.loss_fn = GRPOLoss(config=self.config)
        self.optimizer = AdamW(
            [p for p in self.model.policy_model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )
        self.logger = MetricLogger(log_dir=str(self.output_dir))

    def train_step(self, batch_examples: List[MathExample]) -> dict:
        """Execute a single GRPO step: rollout -> score -> advantage -> policy update."""
        prompts = [ex.prompt for ex in batch_examples]
        ground_truths = [ex.ground_truth for ex in batch_examples]

        # 1. Collect rollouts and verifiable rewards
        rollout_batch = self.rollout_engine.collect_rollouts(
            prompts=prompts, ground_truths=ground_truths
        )

        # 2. Optimization epoch
        self.model.policy_model.train()
        self.optimizer.zero_grad()

        # Compute forward log-probs under current policy (with gradient)
        policy_logps = self.model.get_token_log_probs(
            self.model.policy_model,
            rollout_batch.input_ids,
            rollout_batch.attention_mask,
        )

        loss, metrics = self.loss_fn(
            policy_logps=policy_logps,
            old_policy_logps=rollout_batch.old_policy_logps,
            ref_logps=rollout_batch.ref_logps,
            advantages=rollout_batch.advantages,
            response_mask=rollout_batch.response_mask,
        )

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.policy_model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        # Combine with rollout metrics
        correct_count = sum(1 for res in rollout_batch.reward_results if res.is_correct is True)
        total_count = len(rollout_batch.reward_results)

        step_metrics = {
            **metrics,
            "mean_reward": rollout_batch.raw_rewards.mean().item(),
            "max_reward": rollout_batch.raw_rewards.max().item(),
            "accuracy": (correct_count / total_count) if total_count > 0 else 0.0,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        }

        return step_metrics

    def train(self) -> List[dict]:
        """Run full GRPO training loop."""
        console.rule("[bold cyan]DeepAlign-RL: Starting GRPO Training Loop[/bold cyan]")
        console.print(
            f"[bold]Group Size (G):[/bold] {self.config.num_generations} | "
            f"[bold]Batch Prompts:[/bold] {self.config.batch_size} | "
            f"[bold]Total Steps:[/bold] {self.config.total_steps} | "
            f"[bold]KL Beta:[/bold] {self.config.beta_kl}"
        )

        num_train = len(self.train_dataset)
        history = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            train_task = progress.add_task("[green]GRPO Post-Training...", total=self.config.total_steps)

            for step in range(1, self.config.total_steps + 1):
                # Sample batch of prompts
                batch_indices = [
                    (step * self.config.batch_size + i) % num_train
                    for i in range(self.config.batch_size)
                ]
                batch_examples = [self.train_dataset[idx] for idx in batch_indices]

                metrics = self.train_step(batch_examples)
                history.append(metrics)
                self.logger.log(step=step, metrics=metrics, display=(step % 5 == 0 or step == 1))

                progress.update(train_task, advance=1)

                if step % self.config.save_every == 0 or step == self.config.total_steps:
                    self.save_checkpoint(step)

        console.rule("[bold green]GRPO Training Completed Successfully[/bold green]")
        console.print(self.logger.summary_table("GRPO Final Training Dynamics"))
        return history

    def save_checkpoint(self, step: int):
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Save model state
        torch.save(self.model.policy_model.state_dict(), ckpt_dir / "policy_model.pt")
        console.print(f"[dim]Saved checkpoint to {ckpt_dir}[/dim]")
