"""Command-line interface for DeepAlign-RL engine."""

import argparse
import sys
from pathlib import Path
import torch
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deepalign_rl.core.grpo import compute_group_advantages, GRPOLoss
from deepalign_rl.core.dpo import DPOLoss
from deepalign_rl.datasets.math_reasoning import MathReasoningDataset, CURATED_MATH_PROBLEMS
from deepalign_rl.datasets.dpo_dataset import PreferenceDataset
from deepalign_rl.evaluation.benchmark import ReasoningBenchmark
from deepalign_rl.models.policy import PolicyModelWrapper
from deepalign_rl.rewards.composite import CompositeReward
from deepalign_rl.rewards.format_verifier import FormatReward
from deepalign_rl.rewards.math_verifier import VerifiableMathReward
from deepalign_rl.trainers.grpo_trainer import GRPOTrainer
from deepalign_rl.trainers.dpo_trainer import DPOTrainer
from deepalign_rl.utils.config import GRPOConfig, DPOConfig, ModelConfig, TrainingPipelineConfig
from deepalign_rl.utils.device import get_optimal_device, get_device_name
from deepalign_rl.utils.logger import console


def run_demo():
    """Runs a comprehensive, self-contained terminal demonstration of all core engine components."""
    device = get_optimal_device()

    console.rule("[bold cyan]DeepAlign-RL — Reasoning & Alignment Post-Training Engine[/bold cyan]")
    console.print(
        Panel(
            f"[bold]Active Device:[/bold] {get_device_name(device)}\n"
            f"[bold]PyTorch Version:[/bold] {torch.__version__}\n"
            f"[bold]Architecture:[/bold] DeepSeek-style GRPO (Critic-Free) + DPO + Verifiable Math/Logic Rewards\n"
            f"[bold]Symbolic Verification Engine:[/bold] SymPy v1.14 with LaTeX canonicalization",
            title="[bold green]System Status[/bold green]",
            border_style="cyan",
        )
    )

    # 1. Demonstrate Verifiable Math Reward
    console.rule("[bold magenta]1. Verifiable Math & CoT Format Reward (SymPy-backed)[/bold magenta]")
    math_verifier = VerifiableMathReward(weight=1.0)
    format_verifier = FormatReward(weight=0.5)
    composite = CompositeReward([math_verifier, format_verifier])

    test_prompt = "Solve for x: 3*x + 15 = 48. Show reasoning in <think> and final answer in \\boxed{}."
    target_answer = "11"

    candidates = [
        (
            "Candidate A (Perfect CoT & Correct)",
            "<think>Subtract 15 from both sides: 3*x = 33. Divide by 3: x = 11.</think>\n<answer>\\boxed{11}</answer>",
        ),
        (
            "Candidate B (Symbolic Fraction Equivalence)",
            "<think>3*x = 33 -> x = 33/3 = 11.</think>\n<answer>\\boxed{33/3}</answer>",
        ),
        (
            "Candidate C (Bad Formatting, Correct Answer)",
            "3*x + 15 = 48, so 3*x = 33, x = 11. The answer is 11.",
        ),
        (
            "Candidate D (Arithmetic Hallucination)",
            "<think>Subtract 15 from 48 to get 35. Divide by 3: x = 11.66.</think>\n<answer>\\boxed{11.66}</answer>",
        ),
    ]

    reward_table = Table(title="Verifiable Reward Attribution Breakdown", header_style="bold cyan")
    reward_table.add_column("Candidate", style="bold")
    reward_table.add_column("Math Eq.", justify="center")
    reward_table.add_column("Format Eq.", justify="center")
    reward_table.add_column("Total Reward", justify="right", style="bold green")
    reward_table.add_column("SymPy Proof Explanation")

    scores = []
    for label, text in candidates:
        res = composite.evaluate(test_prompt, text, target_answer)
        scores.append(res.score)
        m_ok = res.details["components"]["verifiable_math"]["explanation"]
        f_ok = res.details["components"]["format_verifier"]["format_valid"]

        math_symbol = "[green]PASS[/green]" if res.is_correct else "[red]FAIL[/red]"
        fmt_symbol = "[green]PASS[/green]" if f_ok else "[red]FAIL[/red]"

        reward_table.add_row(label, math_symbol, fmt_symbol, f"{res.score:.2f}", m_ok)

    console.print(reward_table)

    # 2. Demonstrate GRPO Group-Relative Advantage Normalization
    console.rule("[bold magenta]2. GRPO Group-Relative Advantage Normalization (Critic-Free)[/bold magenta]")
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    group_size = len(scores)
    advantages = compute_group_advantages(scores_tensor, group_size=group_size)

    grpo_table = Table(title="GRPO Group Statistics (G = 4)", header_style="bold cyan")
    grpo_table.add_column("Sample", style="bold")
    grpo_table.add_column("Raw Reward (r_i)", justify="right")
    grpo_table.add_column("Group Mean (μ)", justify="right")
    grpo_table.add_column("Group Std (σ)", justify="right")
    grpo_table.add_column("Standardized Advantage (Â_i)", justify="right", style="bold yellow")
    grpo_table.add_column("Gradient Action")

    mean_r = scores_tensor.mean().item()
    std_r = scores_tensor.std().item()

    for i, (label, _) in enumerate(candidates):
        adv = advantages[i].item()
        if adv > 0.1:
            action = "[green]+ Reinforce tokens (Positive Policy Gradient)[/green]"
        elif adv < -0.1:
            action = "[red]- Suppress tokens (Negative Policy Gradient)[/red]"
        else:
            action = "[dim]0 Neutral[/dim]"

        grpo_table.add_row(
            f"Rollout #{i+1}",
            f"{scores[i]:.2f}",
            f"{mean_r:.2f}",
            f"{std_r:.2f}",
            f"{adv:+.3f}",
            action,
        )

    console.print(grpo_table)

    # 3. Demonstrate DPO Implicit Reward Dynamics
    console.rule("[bold magenta]3. Direct Preference Optimization (DPO) Implicit Rewards[/bold magenta]")
    dpo_loss_fn = DPOLoss(config=DPOConfig(beta=0.1))

    # Synthetic log-prob tensors for chosen vs rejected
    pol_chosen = torch.tensor([-12.5], dtype=torch.float32)
    pol_rejected = torch.tensor([-22.1], dtype=torch.float32)
    ref_chosen = torch.tensor([-13.0], dtype=torch.float32)
    ref_rejected = torch.tensor([-15.5], dtype=torch.float32)

    loss_val, dpo_metrics = dpo_loss_fn(pol_chosen, pol_rejected, ref_chosen, ref_rejected)

    dpo_table = Table(title="DPO Bradley-Terry Optimization Metrics (β = 0.1)", header_style="bold cyan")
    dpo_table.add_column("Metric", style="bold")
    dpo_table.add_column("Value", style="bold green")

    for k, v in dpo_metrics.items():
        dpo_table.add_row(k.replace("_", " ").title(), f"{v:+.4f}" if isinstance(v, float) else str(v))

    console.print(dpo_table)

    console.rule("[bold green]Demonstration Completed Successfully[/bold green]")
    console.print(
        "\n[bold]Next Steps:[/bold]\n"
        "• Launch visualizer UI: [cyan]deepalign serve-ui --port 8000[/cyan]\n"
        "• Run mini GRPO training loop: [cyan]deepalign train-grpo --steps 10 --dummy[/cyan]\n"
        "• Run test suite: [cyan]pytest tests/[/cyan]\n"
    )


def run_train_grpo(args):
    """Run GRPO training."""
    max_resp = 32 if args.dummy else 256
    config = GRPOConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        num_generations=args.group_size,
        max_response_length=max_resp,
    )
    model_cfg = ModelConfig(
        model_name_or_path=args.model,
        is_dummy=args.dummy,
        device=args.device,
    )

    console.print(f"[bold cyan]Initializing Policy Model Wrapper (Dummy: {args.dummy})...[/bold cyan]")
    model_wrapper = PolicyModelWrapper(model_cfg)

    reward_verifier = CompositeReward([
        VerifiableMathReward(weight=1.0),
        FormatReward(weight=0.5),
    ])

    dataset = MathReasoningDataset().get_curated()
    trainer = GRPOTrainer(
        model_wrapper=model_wrapper,
        reward_verifier=reward_verifier,
        config=config,
        train_dataset=dataset,
        output_dir=args.output_dir,
    )

    trainer.train()


def run_train_dpo(args):
    """Run DPO training."""
    config = DPOConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        beta=args.beta,
    )
    model_cfg = ModelConfig(
        model_name_or_path=args.model,
        is_dummy=args.dummy,
        device=args.device,
    )

    model_wrapper = PolicyModelWrapper(model_cfg)
    dataset = PreferenceDataset().get_all()

    trainer = DPOTrainer(
        model_wrapper=model_wrapper,
        config=config,
        train_dataset=dataset,
        output_dir=args.output_dir,
    )

    trainer.train()


def run_serve_ui(args):
    """Start the FastAPI visualizer server."""
    console.rule("[bold cyan]Starting DeepAlign-RL Visualizer Dashboard[/bold cyan]")
    console.print(f"Open browser at: [bold green]http://{args.host}:{args.port}[/bold green]")
    uvicorn.run("deepalign_rl.visualizer.server:app", host=args.host, port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        prog="deepalign",
        description="DeepAlign-RL: Reasoning & Alignment Post-Training Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # demo
    subparsers.add_parser("demo", help="Run end-to-end interactive CLI demonstration")

    # train-grpo
    p_grpo = subparsers.add_parser("train-grpo", help="Train policy via Group Relative Policy Optimization")
    p_grpo.add_argument("--steps", type=int, default=20, help="Number of training steps")
    p_grpo.add_argument("--batch-size", type=int, default=2, help="Prompt batch size")
    p_grpo.add_argument("--group-size", type=int, default=4, help="Rollout group size G")
    p_grpo.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    p_grpo.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu)")
    p_grpo.add_argument("--dummy", action="store_true", help="Use lightweight mock model for fast dry runs")
    p_grpo.add_argument("--output-dir", type=str, default="./outputs/grpo", help="Output directory")

    # train-dpo
    p_dpo = subparsers.add_parser("train-dpo", help="Train policy via Direct Preference Optimization")
    p_dpo.add_argument("--steps", type=int, default=20, help="Number of training steps")
    p_dpo.add_argument("--batch-size", type=int, default=2, help="Preference pair batch size")
    p_dpo.add_argument("--beta", type=float, default=0.1, help="DPO beta temperature")
    p_dpo.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    p_dpo.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu)")
    p_dpo.add_argument("--dummy", action="store_true", help="Use lightweight mock model for fast dry runs")
    p_dpo.add_argument("--output-dir", type=str, default="./outputs/dpo", help="Output directory")

    # serve-ui
    p_ui = subparsers.add_parser("serve-ui", help="Launch interactive visualizer web dashboard")
    p_ui.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    p_ui.add_argument("--port", type=int, default=8000, help="Port")

    args = parser.parse_args()

    if args.command == "demo" or args.command is None:
        run_demo()
    elif args.command == "train-grpo":
        run_train_grpo(args)
    elif args.command == "train-dpo":
        run_train_dpo(args)
    elif args.command == "serve-ui":
        run_serve_ui(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
