"""Full Training Loop and Engine for Direct Preference Optimization (DPO)."""

from pathlib import Path
from typing import List, Optional
import torch
from torch.optim import AdamW
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from deepalign_rl.core.dpo import DPOLoss
from deepalign_rl.datasets.dpo_dataset import PreferenceExample
from deepalign_rl.models.policy import PolicyModelWrapper
from deepalign_rl.utils.config import DPOConfig
from deepalign_rl.utils.logger import MetricLogger, console


class DPOTrainer:
    """Orchestrates Direct Preference Optimization on pairwise reasoning trajectories."""

    def __init__(
        self,
        model_wrapper: PolicyModelWrapper,
        config: DPOConfig,
        train_dataset: List[PreferenceExample],
        val_dataset: Optional[List[PreferenceExample]] = None,
        output_dir: str = "./outputs/dpo",
    ):
        self.model = model_wrapper
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset or []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = DPOLoss(config=self.config)
        self.optimizer = AdamW(
            [p for p in self.model.policy_model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )
        self.logger = MetricLogger(log_dir=str(self.output_dir))

    def _prepare_sequence_logps(
        self,
        model: torch.nn.Module,
        prompts: List[str],
        completions: List[str],
    ) -> torch.Tensor:
        """Compute sum of log probabilities of the completion tokens."""
        full_texts = [f"{p} {c}" for p, c in zip(prompts, completions)]

        if hasattr(self.model.tokenizer, "batch_encode"):
            encoded = self.model.tokenizer.batch_encode(full_texts, padding=True)
            p_encoded = self.model.tokenizer.batch_encode(prompts, padding=True)
            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded["attention_mask"].to(self.model.device)
            p_lengths = [int(m.sum().item()) for m in p_encoded["attention_mask"]]
        else:
            encoded = self.model.tokenizer(
                full_texts, return_tensors="pt", padding=True, truncation=True
            )
            p_encoded = self.model.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded["attention_mask"].to(self.model.device)
            p_lengths = [int(m.sum().item()) for m in p_encoded["attention_mask"]]

        token_logps = self.model.get_token_log_probs(model, input_ids, attention_mask)
        # token_logps is [B, S-1]

        # Mask only the completion tokens
        seq_logps = []
        for i in range(len(prompts)):
            p_len = p_lengths[i]
            total_len = int(attention_mask[i].sum().item())
            start_idx = max(0, p_len - 1)
            end_idx = max(0, total_len - 1)
            if end_idx > start_idx:
                comp_logps = token_logps[i, start_idx:end_idx].sum()
            else:
                comp_logps = torch.tensor(0.0, device=self.model.device)
            seq_logps.append(comp_logps)

        return torch.stack(seq_logps)

    def train_step(self, batch_examples: List[PreferenceExample]) -> dict:
        prompts = [ex.prompt for ex in batch_examples]
        chosen = [ex.chosen for ex in batch_examples]
        rejected = [ex.rejected for ex in batch_examples]

        self.model.policy_model.train()
        self.optimizer.zero_grad()

        # 1. Policy log-probs
        policy_chosen_logps = self._prepare_sequence_logps(
            self.model.policy_model, prompts, chosen
        )
        policy_rejected_logps = self._prepare_sequence_logps(
            self.model.policy_model, prompts, rejected
        )

        # 2. Reference log-probs (no gradient)
        with torch.no_grad():
            ref_chosen_logps = self._prepare_sequence_logps(
                self.model.ref_model, prompts, chosen
            )
            ref_rejected_logps = self._prepare_sequence_logps(
                self.model.ref_model, prompts, rejected
            )

        # 3. DPO Loss computation
        loss, metrics = self.loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.policy_model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return {
            **metrics,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        }

    def train(self) -> List[dict]:
        """Run DPO training loop."""
        console.rule("[bold cyan]DeepAlign-RL: Starting DPO Training Loop[/bold cyan]")
        console.print(
            f"[bold]DPO Beta:[/bold] {self.config.beta} | "
            f"[bold]Batch Size:[/bold] {self.config.batch_size} | "
            f"[bold]Total Steps:[/bold] {self.config.total_steps}"
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
            task = progress.add_task("[magenta]DPO Preference Optimization...", total=self.config.total_steps)

            for step in range(1, self.config.total_steps + 1):
                batch_indices = [
                    (step * self.config.batch_size + i) % num_train
                    for i in range(self.config.batch_size)
                ]
                batch_examples = [self.train_dataset[idx] for idx in batch_indices]

                metrics = self.train_step(batch_examples)
                history.append(metrics)
                self.logger.log(step=step, metrics=metrics, display=(step % 5 == 0 or step == 1))

                progress.update(task, advance=1)

        console.rule("[bold green]DPO Training Completed Successfully[/bold green]")
        console.print(self.logger.summary_table("DPO Final Training Dynamics"))
        return history
