"""Evaluation Benchmark Suite for Reasoning Models."""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from rich.table import Table

from deepalign_rl.datasets.math_reasoning import MathExample
from deepalign_rl.models.policy import PolicyModelWrapper
from deepalign_rl.rewards.composite import CompositeReward
from deepalign_rl.rewards.format_verifier import FormatReward
from deepalign_rl.rewards.math_verifier import VerifiableMathReward
from deepalign_rl.utils.logger import console


class BenchmarkResult(BaseModel):
    total_samples: int
    accuracy: float
    format_compliance: float
    avg_reasoning_length: float
    avg_total_length: float
    pass_at_k: Optional[float] = None
    detailed_results: List[Dict[str, Any]] = Field(default_factory=list)


class ReasoningBenchmark:
    """Evaluates mathematical reasoning accuracy, format adherence, and reasoning depth."""

    def __init__(self, reward_verifier: Optional[CompositeReward] = None):
        if reward_verifier is None:
            self.reward_verifier = CompositeReward([
                VerifiableMathReward(weight=1.0),
                FormatReward(weight=0.5),
            ])
        else:
            self.reward_verifier = reward_verifier

    def run_evaluation(
        self,
        model_wrapper: PolicyModelWrapper,
        examples: List[MathExample],
        num_samples_per_prompt: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> BenchmarkResult:
        console.rule(f"[bold cyan]Running Reasoning Benchmark ({len(examples)} problems)[/bold cyan]")

        prompts = [ex.prompt for ex in examples]
        ground_truths = [ex.ground_truth for ex in examples]

        all_completions, _, _, _ = model_wrapper.generate_responses(
            prompt_texts=prompts,
            num_generations=num_samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        detailed = []
        correct_count = 0
        format_count = 0
        think_lengths = []
        total_lengths = []

        for i, ex in enumerate(examples):
            # completions slice for this example
            start_i = i * num_samples_per_prompt
            comp_slice = all_completions[start_i : start_i + num_samples_per_prompt]

            sample_correct = False
            for c in comp_slice:
                res = self.reward_verifier.evaluate(ex.prompt, c, ex.ground_truth)
                is_math_correct = res.details.get("components", {}).get("verifiable_math", {}).get("explanation", "").startswith("Exact") or \
                                  res.details.get("components", {}).get("verifiable_math", {}).get("explanation", "").startswith("Numerical") or \
                                  res.details.get("components", {}).get("verifiable_math", {}).get("explanation", "").startswith("SymPy")
                is_fmt_correct = res.details.get("components", {}).get("format_verifier", {}).get("format_valid", False)
                think_len = res.details.get("components", {}).get("format_verifier", {}).get("think_length", 0)

                if is_math_correct:
                    sample_correct = True
                    correct_count += 1
                if is_fmt_correct:
                    format_count += 1

                think_lengths.append(think_len)
                total_lengths.append(len(c))

                detailed.append({
                    "id": ex.id,
                    "prompt": ex.prompt,
                    "ground_truth": ex.ground_truth,
                    "completion": c,
                    "math_correct": is_math_correct,
                    "format_valid": is_fmt_correct,
                    "reward_score": res.score,
                    "details": res.details,
                })

        total_tested = len(examples) * num_samples_per_prompt
        acc = (correct_count / total_tested) if total_tested > 0 else 0.0
        fmt = (format_count / total_tested) if total_tested > 0 else 0.0
        avg_think = float(np.mean(think_lengths)) if think_lengths else 0.0
        avg_tot = float(np.mean(total_lengths)) if total_lengths else 0.0

        result = BenchmarkResult(
            total_samples=total_tested,
            accuracy=acc,
            format_compliance=fmt,
            avg_reasoning_length=avg_think,
            avg_total_length=avg_tot,
            detailed_results=detailed,
        )

        # Print summary table
        table = Table(title="Reasoning Benchmark Results", header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")

        table.add_row("Total Problems", str(len(examples)))
        table.add_row("Samples per Prompt", str(num_samples_per_prompt))
        table.add_row("Math Accuracy (Verified by SymPy)", f"{acc * 100:.2f}%")
        table.add_row("Format Compliance (<think> & <answer>)", f"{fmt * 100:.2f}%")
        table.add_row("Avg Reasoning Characters", f"{avg_think:.1f}")
        table.add_row("Avg Total Output Length", f"{avg_tot:.1f}")

        console.print(table)
        return result
