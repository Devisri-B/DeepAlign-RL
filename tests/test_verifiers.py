"""Unit tests for verifiable reward functions."""

import pytest
from deepalign_rl.rewards.math_verifier import VerifiableMathReward
from deepalign_rl.rewards.format_verifier import FormatReward
from deepalign_rl.rewards.composite import CompositeReward


def test_math_extraction_and_equivalence():
    verifier = VerifiableMathReward(weight=1.0)

    # 1. Exact integer in \boxed{}
    res1 = verifier.evaluate(
        prompt="Solve 2+2",
        completion="<think>Simple addition</think>\n<answer>\\boxed{4}</answer>",
        ground_truth="4",
    )
    assert res1.is_correct is True
    assert res1.score == 1.0

    # 2. Fraction equivalence (1/2 vs 0.5)
    res2 = verifier.evaluate(
        prompt="Fraction test",
        completion="The result is \\boxed{\\frac{1}{2}}",
        ground_truth="0.5",
    )
    assert res2.is_correct is True
    assert res2.score == 1.0

    # 3. GSM8K #### style
    res3 = verifier.evaluate(
        prompt="GSM test",
        completion="Step by step calculation.\n#### 42",
        ground_truth="42",
    )
    assert res3.is_correct is True

    # 4. Symbolic equivalence (3*x vs x*3)
    res4 = verifier.evaluate(
        prompt="Algebra test",
        completion="Answer is \\boxed{3*x + 1}",
        ground_truth="1 + 3*x",
    )
    assert res4.is_correct is True

    # 5. Incorrect answer
    res5 = verifier.evaluate(
        prompt="Mismatch test",
        completion="\\boxed{99}",
        ground_truth="100",
    )
    assert res5.is_correct is False
    assert res5.score == 0.0


def test_format_verifier():
    verifier = FormatReward(weight=0.5, min_think_chars=10)

    # Well-formatted CoT
    good_text = "<think>Here is detailed thinking with sufficient characters</think>\n<answer>42</answer>"
    res_good = verifier.evaluate(prompt="dummy", completion=good_text)
    assert res_good.is_correct is True
    assert res_good.score == 0.5

    # Unclosed think tag
    bad_text = "<think>I forgot to close my thought"
    res_bad = verifier.evaluate(prompt="dummy", completion=bad_text)
    assert res_bad.score < 0.2


def test_composite_reward():
    math_v = VerifiableMathReward(weight=1.0)
    fmt_v = FormatReward(weight=0.5)
    composite = CompositeReward([math_v, fmt_v])

    text = "<think>Calculating the product of 2 and 5 gives exactly 10.</think>\n<answer>\\boxed{10}</answer>"
    res = composite.evaluate(prompt="2*5", completion=text, ground_truth="10")
    assert res.score == 1.5
    assert "verifiable_math" in res.details["breakdown"]
    assert "format_verifier" in res.details["breakdown"]
