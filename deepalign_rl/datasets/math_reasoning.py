"""Curated and parametric verifiable math reasoning dataset."""

import random
from typing import Dict, List, Optional
from pydantic import BaseModel


class MathExample(BaseModel):
    id: str
    prompt: str
    ground_truth: str
    target_solution: str
    category: str


# Hand-crafted challenging GSM8K/MATH style reasoning problems
CURATED_MATH_PROBLEMS: List[Dict[str, str]] = [
    {
        "id": "math_001",
        "prompt": "Solve for x in the equation: 3*x + 15 = 48. What is the value of x? Show your reasoning inside <think> tags and put your final answer inside \\boxed{}.",
        "ground_truth": "11",
        "target_solution": "<think>Subtract 15 from both sides: 3*x = 33. Divide by 3: x = 11.</think>\n<answer>\\boxed{11}</answer>",
        "category": "algebra",
    },
    {
        "id": "math_002",
        "prompt": "A store sells notebooks for $4 each and pens for $1.50 each. Maya buys 6 notebooks and 8 pens. If she pays with a $50 bill, how much change does she receive? Show reasoning inside <think> and final answer in \\boxed{}.",
        "ground_truth": "14",
        "target_solution": "<think>Cost of notebooks = 6 * 4 = 24. Cost of pens = 8 * 1.50 = 12. Total cost = 24 + 12 = 36. Change = 50 - 36 = 14.</think>\n<answer>\\boxed{14}</answer>",
        "category": "arithmetic",
    },
    {
        "id": "math_003",
        "prompt": "What is the positive solution to the equation x^2 - 7*x + 10 = 0? Provide your reasoning inside <think> and final answer inside \\boxed{}.",
        "ground_truth": "5",
        "target_solution": "<think>Factor the quadratic: (x - 2)(x - 5) = 0. The solutions are x = 2 and x = 5. The larger positive solution is 5.</think>\n<answer>\\boxed{5}</answer>",
        "category": "algebra",
    },
    {
        "id": "math_004",
        "prompt": "Evaluate the fraction: (3/4) + (2/5). Express your answer as a simplified fraction or decimal. Provide reasoning in <think> and final answer in \\boxed{}.",
        "ground_truth": "23/20",
        "target_solution": "<think>Common denominator for 4 and 5 is 20. 3/4 = 15/20. 2/5 = 8/20. Sum = 15/20 + 8/20 = 23/20 = 1.15.</think>\n<answer>\\boxed{23/20}</answer>",
        "category": "fractions",
    },
    {
        "id": "math_005",
        "prompt": "A train travels at 60 mph for 2.5 hours, and then 45 mph for 3 hours. What is the total distance traveled in miles? Show reasoning in <think> and final answer in \\boxed{}.",
        "ground_truth": "285",
        "target_solution": "<think>Distance 1 = 60 * 2.5 = 150 miles. Distance 2 = 45 * 3 = 135 miles. Total distance = 150 + 135 = 285 miles.</think>\n<answer>\\boxed{285}</answer>",
        "category": "rate_distance",
    },
    {
        "id": "math_006",
        "prompt": "If f(x) = 2*x^2 - 3*x + 7, what is the value of f(4)? Show your reasoning inside <think> and answer in \\boxed{}.",
        "ground_truth": "27",
        "target_solution": "<think>f(4) = 2*(4)^2 - 3*(4) + 7 = 2*16 - 12 + 7 = 32 - 12 + 7 = 27.</think>\n<answer>\\boxed{27}</answer>",
        "category": "functions",
    },
    {
        "id": "math_007",
        "prompt": "A circle has a circumference of 18*pi. What is the area of the circle in terms of pi? (e.g., if 25*pi, enter 81*pi). Show reasoning in <think> and final answer in \\boxed{}.",
        "ground_truth": "81*pi",
        "target_solution": "<think>Circumference C = 2*pi*r = 18*pi -> r = 9. Area A = pi*r^2 = pi*(9^2) = 81*pi.</think>\n<answer>\\boxed{81*pi}</answer>",
        "category": "geometry",
    },
    {
        "id": "math_008",
        "prompt": "In a class of 40 students, 65% passed the math exam. How many students failed the exam? Show your reasoning inside <think> and final answer inside \\boxed{}.",
        "ground_truth": "14",
        "target_solution": "<think>If 65% passed, 35% failed. 35% of 40 = 0.35 * 40 = 14 students.</think>\n<answer>\\boxed{14}</answer>",
        "category": "percentages",
    },
]


class MathReasoningDataset:
    """Dataset generator supporting curated benchmarks and procedural parametric generation."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.curated = [MathExample(**item) for item in CURATED_MATH_PROBLEMS]

    def get_curated(self) -> List[MathExample]:
        return list(self.curated)

    def generate_parametric(self, count: int = 50) -> List[MathExample]:
        """Procedurally generate arithmetic and algebraic reasoning problems."""
        examples = []
        for i in range(count):
            prob_type = self.rng.choice(["linear", "word_problem", "percentage", "system"])
            if prob_type == "linear":
                # a*x + b = c
                a = self.rng.randint(2, 9)
                x = self.rng.randint(2, 20)
                b = self.rng.randint(5, 50)
                c = a * x + b
                prompt = (
                    f"Solve for x: {a}*x + {b} = {c}. "
                    "Write your step-by-step reasoning inside <think>...</think> and your final answer inside \\boxed{}."
                )
                solution = (
                    f"<think>Subtract {b} from {c}: {a}*x = {c - b}. "
                    f"Divide by {a}: x = {x}.</think>\n<answer>\\boxed{{{x}}}</answer>"
                )
                examples.append(
                    MathExample(
                        id=f"param_linear_{i}",
                        prompt=prompt,
                        ground_truth=str(x),
                        target_solution=solution,
                        category="linear_equation",
                    )
                )
            elif prob_type == "percentage":
                pct = self.rng.choice([10, 15, 20, 25, 30, 40, 50])
                base = self.rng.choice([50, 80, 100, 120, 150, 200, 240, 300])
                val = int(base * pct / 100)
                prompt = (
                    f"What is {pct}% of {base}? "
                    "Write your step-by-step reasoning inside <think>...</think> and your final answer inside \\boxed{}."
                )
                solution = (
                    f"<think>{pct}% is {pct}/100 = {pct/100:.2f}. "
                    f"Multiply by {base}: {pct/100:.2f} * {base} = {val}.</think>\n<answer>\\boxed{{{val}}}</answer>"
                )
                examples.append(
                    MathExample(
                        id=f"param_pct_{i}",
                        prompt=prompt,
                        ground_truth=str(val),
                        target_solution=solution,
                        category="percentage",
                    )
                )
            elif prob_type == "word_problem":
                apples = self.rng.randint(3, 12)
                price = self.rng.randint(2, 6)
                total = apples * price
                prompt = (
                    f"Liam bought {apples} baskets of fruit. Each basket costs ${price}. "
                    "How much did Liam spend in total? "
                    "Show reasoning inside <think>...</think> and write final answer in \\boxed{}."
                )
                solution = (
                    f"<think>Total cost = number of baskets * cost per basket = {apples} * {price} = {total}.</think>\n"
                    f"<answer>\\boxed{{{total}}}</answer>"
                )
                examples.append(
                    MathExample(
                        id=f"param_word_{i}",
                        prompt=prompt,
                        ground_truth=str(total),
                        target_solution=solution,
                        category="word_problem",
                    )
                )
            else:
                # Difference of squares: (a+b)(a-b)
                a = self.rng.randint(10, 30)
                b = self.rng.randint(1, 9)
                ans = a**2 - b**2
                prompt = (
                    f"Calculate ({a} + {b}) * ({a} - {b}). "
                    "Show your reasoning inside <think>...</think> and your final answer in \\boxed{}."
                )
                solution = (
                    f"<think>Using difference of squares: ({a}+{b})*({a}-{b}) = {a}^2 - {b}^2 = {a**2} - {b**2} = {ans}.</think>\n"
                    f"<answer>\\boxed{{{ans}}}</answer>"
                )
                examples.append(
                    MathExample(
                        id=f"param_diff_{i}",
                        prompt=prompt,
                        ground_truth=str(ans),
                        target_solution=solution,
                        category="difference_squares",
                    )
                )

        return examples
