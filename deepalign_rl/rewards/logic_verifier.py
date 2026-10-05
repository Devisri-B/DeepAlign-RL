"""Verifiable Logic & Code Sandbox Reward Verifier.

Executes code solutions or evaluates logic constraint satisfaction
against verifiable test assertions in a restricted execution environment.
"""

import ast
import multiprocessing
import traceback
from typing import Any, Dict, List, Optional
from deepalign_rl.rewards.base import BaseRewardVerifier, RewardResult


def _worker_exec(code: str, test_cases: List[str], queue: multiprocessing.Queue):
    """Executes code and test cases in a subprocess for safety and timeout isolation."""
    scope: Dict[str, Any] = {}
    try:
        # Check AST for hazardous imports
        parsed = ast.parse(code)
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Disallow os, subprocess, sys, socket for safety
                module_name = getattr(node, "module", "") or ""
                names = [alias.name for alias in getattr(node, "names", [])]
                for bad in ["os", "subprocess", "sys", "socket", "shutil", "builtins"]:
                    if bad in module_name or bad in names:
                        queue.put((False, f"Restricted import detected: {bad}", 0, len(test_cases)))
                        return

        exec(code, scope)
        passed = 0
        for test in test_cases:
            try:
                exec(test, scope)
                passed += 1
            except Exception as e:
                queue.put((False, f"Test failed: {test} -> {str(e)}", passed, len(test_cases)))
                return

        queue.put((True, "All tests passed", passed, len(test_cases)))
    except Exception as e:
        queue.put((False, f"Execution error: {str(e)}\n{traceback.format_exc()}", 0, len(test_cases)))


class VerifiableLogicReward(BaseRewardVerifier):
    """Evaluates logic and code solutions against verifiable test suites with timeouts."""

    def __init__(
        self,
        name: str = "logic_verifier",
        weight: float = 1.0,
        timeout_seconds: float = 2.0,
    ):
        super().__init__(name=name, weight=weight)
        self.timeout_seconds = timeout_seconds

    def extract_code(self, text: str) -> Optional[str]:
        """Extract Python code blocks from completion."""
        import re
        blocks = re.findall(r"```(?:python)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            return blocks[-1].strip()
        # Fallback: check if entire completion parses as valid python
        try:
            ast.parse(text)
            return text.strip()
        except Exception:
            return None

    def evaluate(self, prompt: str, completion: str, ground_truth: Optional[str] = None) -> RewardResult:
        """Evaluate completion against test cases in ground_truth.

        ground_truth can be formatted as lines of assert statements or python test code.
        """
        code = self.extract_code(completion)
        if not code:
            return RewardResult(
                score=0.0,
                is_correct=False,
                details={"error": "No valid Python code found in completion"},
                verifier_name=self.name,
            )

        test_cases = []
        if ground_truth:
            test_cases = [line.strip() for line in ground_truth.strip().split("\n") if line.strip()]

        if not test_cases:
            # Syntax validation only
            try:
                ast.parse(code)
                return RewardResult(
                    score=1.0 * self.weight,
                    is_correct=True,
                    details={"syntax_valid": True, "note": "No test cases provided, syntax verified"},
                    verifier_name=self.name,
                )
            except Exception as e:
                return RewardResult(
                    score=0.0,
                    is_correct=False,
                    details={"syntax_error": str(e)},
                    verifier_name=self.name,
                )

        # Run with timeout sandbox
        queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_worker_exec, args=(code, test_cases, queue))
        proc.start()
        proc.join(timeout=self.timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.2)
            return RewardResult(
                score=0.0,
                is_correct=False,
                details={"error": f"Execution timed out after {self.timeout_seconds}s"},
                verifier_name=self.name,
            )

        if not queue.empty():
            success, msg, passed, total = queue.get()
            score = (passed / max(1, total)) if total > 0 else (1.0 if success else 0.0)
            return RewardResult(
                score=score * self.weight,
                is_correct=success,
                details={"message": msg, "passed": passed, "total": total},
                verifier_name=self.name,
            )

        return RewardResult(
            score=0.0,
            is_correct=False,
            details={"error": "Execution process returned no status"},
            verifier_name=self.name,
        )
