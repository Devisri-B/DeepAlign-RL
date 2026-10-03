"""Verifiable Math Reward Function using SymPy and regex parsing.

Validates mathematical reasoning outputs against ground truth answers with
symbolic equivalence, LaTeX canonicalization, and numerical tolerances.
Similar to the rule-based verification in DeepSeekMath and DeepSeek-R1.
"""

import re
from typing import Optional, Tuple
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from deepalign_rl.rewards.base import BaseRewardVerifier, RewardResult


class VerifiableMathReward(BaseRewardVerifier):
    """Verifies mathematical answers using LaTeX parsing and SymPy symbolic equivalence."""

    def __init__(
        self,
        name: str = "verifiable_math",
        weight: float = 1.0,
        tolerance: float = 1e-4,
        allow_symbolic: bool = True,
        strict_format: bool = False,
    ):
        super().__init__(name=name, weight=weight)
        self.tolerance = tolerance
        self.allow_symbolic = allow_symbolic
        self.strict_format = strict_format

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract candidate answer from reasoning text.

        Prioritizes:
        1. <answer> ... </answer>
        2. \\boxed{ ... }
        3. #### ... (GSM8K style)
        4. "The answer is ..." / "Final Answer: ..."
        5. Last line / expression
        """
        if not text:
            return None

        # 1. XML <answer> tag
        xml_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if xml_match:
            candidate = xml_match.group(1).strip()
            # If inner contains boxed, unpack it
            boxed = self._extract_boxed(candidate)
            return boxed if boxed else candidate

        # 2. LaTeX \boxed{...} (supports nested braces)
        boxed = self._extract_boxed(text)
        if boxed:
            return boxed

        # 3. GSM8K "####" format
        gsm_match = re.search(r"####\s*([^\n\r]+)", text)
        if gsm_match:
            return gsm_match.group(1).strip()

        # 4. "The answer is" / "Final Answer:"
        final_patterns = [
            r"(?:the\s+answer\s+is|final\s+answer(?:\s+is)?[:\s]*)\s*([^\n\r.]+)",
            r"(?:answer\s*=\s*)([^\n\r.]+)",
        ]
        for pat in final_patterns:
            match = re.search(pat, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                boxed = self._extract_boxed(candidate)
                return boxed if boxed else candidate

        if self.strict_format:
            return None

        # 5. Fallback: last non-empty line
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if lines:
            last_line = lines[-1]
            last_line = re.sub(r"[.?!]+$", "", last_line).strip()
            # If line has typical natural language words, try extracting the final math/numeric token
            has_words = bool(re.search(r"\b(is|the|answer|equals|result|so|therefore|then)\b", last_line, flags=re.IGNORECASE))
            if has_words:
                num_match = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", last_line)
                if num_match:
                    return num_match[-1]
            return last_line

        return None

    def _extract_boxed(self, text: str) -> Optional[str]:
        """Extract content inside \\boxed{...} handling nested braces."""
        idx = text.rfind(r"\boxed{")
        if idx == -1:
            return None

        start_idx = idx + len(r"\boxed{")
        depth = 1
        curr = start_idx
        while curr < len(text) and depth > 0:
            if text[curr] == "{":
                depth += 1
            elif text[curr] == "}":
                depth -= 1
            curr += 1

        if depth == 0:
            return text[start_idx : curr - 1].strip()
        return None

    def clean_latex(self, text: str) -> str:
        """Normalize LaTeX expressions to standard algebraic strings."""
        if not text:
            return ""

        s = text.strip()
        # Remove markdown/math delimiters
        s = re.sub(r"^\$+|\$+$", "", s)
        s = re.sub(r"^\\\(|\\\)$", "", s)
        s = re.sub(r"^\\\[|\\\]$", "", s)

        # Normalize fractions: \frac{a}{b} -> (a)/(b)
        # Use regex loop for nested \frac
        while r"\frac" in s:
            match = re.search(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", s)
            if not match:
                break
            num, denom = match.groups()
            s = s[: match.start()] + f"(({num})/({denom}))" + s[match.end() :]

        # Normalize multipliers
        s = s.replace(r"\times", "*").replace(r"\cdot", "*")
        # Normalize degrees, percentages, dollar signs
        s = s.replace(r"\%", "%").replace("$", "").replace(",", "")
        if s.endswith("%"):
            try:
                num = float(s[:-1])
                s = str(num / 100.0)
            except ValueError:
                pass

        # Normalize spaces
        s = re.sub(r"\s+", "", s)
        return s

    def check_equivalence(self, pred_raw: str, target_raw: str) -> Tuple[bool, str]:
        """Check if predicted answer is mathematically equivalent to target.

        Uses:
        1. Exact string match after cleaning
        2. Numerical equivalence (float with tolerance)
        3. SymPy symbolic expression simplification
        """
        if not pred_raw or not target_raw:
            return False, "Empty prediction or target"

        pred_clean = self.clean_latex(pred_raw)
        target_clean = self.clean_latex(target_raw)

        # 1. Exact string match
        if pred_clean == target_clean:
            return True, "Exact string match"

        # 2. Float numerical check
        try:
            pred_val = float(pred_clean)
            target_val = float(target_clean)
            if abs(pred_val - target_val) <= max(self.tolerance, abs(target_val) * self.tolerance):
                return True, f"Numerical match: {pred_val} ~ {target_val}"
        except (ValueError, TypeError):
            pass

        # 3. SymPy symbolic check
        if self.allow_symbolic:
            transformations = standard_transformations + (implicit_multiplication_application,)
            try:
                p_expr = parse_expr(pred_clean, transformations=transformations, evaluate=True)
                t_expr = parse_expr(target_clean, transformations=transformations, evaluate=True)

                diff = sympy.simplify(p_expr - t_expr)
                if diff == 0:
                    return True, "SymPy symbolic equivalence"

                # Check float evaluation of expressions
                try:
                    p_num = float(p_expr.evalf())
                    t_num = float(t_expr.evalf())
                    if abs(p_num - t_num) <= max(self.tolerance, abs(t_num) * self.tolerance):
                        return True, f"SymPy numerical evalf match: {p_num} ~ {t_num}"
                except Exception:
                    pass
            except Exception:
                pass

        return False, f"Mismatch: pred='{pred_clean}' vs target='{target_clean}'"

    def evaluate(self, prompt: str, completion: str, ground_truth: Optional[str] = None) -> RewardResult:
        """Evaluate mathematical completion against ground truth."""
        if ground_truth is None:
            return RewardResult(
                score=0.0,
                is_correct=False,
                details={"error": "Ground truth answer is required for math verification"},
                verifier_name=self.name,
            )

        extracted_pred = self.extract_answer(completion)
        boxed_gt = self._extract_boxed(ground_truth)
        if boxed_gt:
            extracted_gt = boxed_gt
        elif "####" in ground_truth:
            extracted_gt = self.extract_answer(ground_truth) or ground_truth.strip()
        else:
            extracted_gt = ground_truth.strip()

        if extracted_pred is None:
            return RewardResult(
                score=0.0,
                is_correct=False,
                details={
                    "extracted_pred": None,
                    "ground_truth": extracted_gt,
                    "reason": "Failed to extract candidate answer from response",
                },
                verifier_name=self.name,
            )

        is_equiv, explanation = self.check_equivalence(extracted_pred, extracted_gt)
        score = 1.0 if is_equiv else 0.0

        return RewardResult(
            score=score * self.weight,
            is_correct=is_equiv,
            details={
                "extracted_pred": extracted_pred,
                "ground_truth": extracted_gt,
                "explanation": explanation,
            },
            verifier_name=self.name,
        )
