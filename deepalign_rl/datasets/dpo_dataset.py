"""Paired preference dataset for Direct Preference Optimization (DPO)."""

from typing import Dict, List
from pydantic import BaseModel


class PreferenceExample(BaseModel):
    id: str
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, str] = {}


CURATED_DPO_PAIRS: List[Dict[str, str]] = [
    {
        "id": "dpo_001",
        "prompt": "Solve for x: 5*x - 10 = 25. Show reasoning in <think> and final answer in \\boxed{}.",
        "chosen": "<think>Step 1: Add 10 to both sides: 5*x = 35.\nStep 2: Divide both sides by 5: x = 7.\nVerify: 5(7) - 10 = 35 - 10 = 25, which matches.</think>\n<answer>\\boxed{7}</answer>",
        "rejected": "The answer is obviously 3. 5*3 = 15 - 10 = 5, wait no, 25 minus 10 is 15 so x = 3.\n\\boxed{3}",
    },
    {
        "id": "dpo_002",
        "prompt": "A bakery makes 48 cupcakes. 3/8 of them are vanilla, and the rest are chocolate. How many chocolate cupcakes are there? Show reasoning in <think> and answer in \\boxed{}.",
        "chosen": "<think>Total cupcakes = 48.\nFraction chocolate = 1 - 3/8 = 5/8.\nNumber of chocolate cupcakes = 48 * (5/8) = (48/8) * 5 = 6 * 5 = 30.\nCheck: Vanilla = 48 * 3/8 = 18. 18 + 30 = 48.</think>\n<answer>\\boxed{30}</answer>",
        "rejected": "<think>48 times 3 is 144, divided by 8 is 18. So the answer is 18.</think>\n<answer>\\boxed{18}</answer>",
    },
    {
        "id": "dpo_003",
        "prompt": "What is 15% of 240? Provide reasoning inside <think> and final answer inside \\boxed{}.",
        "chosen": "<think>10% of 240 is 24.\n5% of 240 is half of 24, which is 12.\nTherefore, 15% of 240 = 24 + 12 = 36.</think>\n<answer>\\boxed{36}</answer>",
        "rejected": "<think>15% means 0.15. 0.15 * 240. Let me guess around 40.</think>\n\\boxed{40}",
    },
    {
        "id": "dpo_004",
        "prompt": "If a right triangle has legs of lengths 9 and 12, what is the length of the hypotenuse? Show reasoning in <think> and final answer in \\boxed{}.",
        "chosen": "<think>By the Pythagorean theorem: c^2 = a^2 + b^2.\nHere a = 9 and b = 12.\na^2 = 81, b^2 = 144.\nc^2 = 81 + 144 = 225.\nc = sqrt(225) = 15.</think>\n<answer>\\boxed{15}</answer>",
        "rejected": "<think>Legs are 9 and 12. Add them together: 9 + 12 = 21.</think>\n<answer>\\boxed{21}</answer>",
    },
]


class PreferenceDataset:
    """Provides curated and generated preference pairs for DPO training."""

    def __init__(self):
        self.examples = [PreferenceExample(**item) for item in CURATED_DPO_PAIRS]

    def get_all(self) -> List[PreferenceExample]:
        return list(self.examples)
