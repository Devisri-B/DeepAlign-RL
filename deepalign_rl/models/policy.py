"""Policy and Reference Model Wrapper for Post-Training."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepalign_rl.models.dummy_model import DummyCausalLM, DummyTokenizer
from deepalign_rl.utils.config import ModelConfig
from deepalign_rl.utils.device import get_optimal_device, get_device_name


class PolicyModelWrapper:
    """Encapsulates Policy model and frozen Reference model for RL / DPO."""

    def __init__(
        self,
        config: ModelConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        if config.is_dummy:
            self.device = torch.device("cpu")
        else:
            self.device = device or (
                torch.device(config.device) if config.device else get_optimal_device()
            )
        self._init_models()

    def _init_models(self):
        if self.config.is_dummy:
            self.tokenizer = DummyTokenizer()
            self.policy_model = DummyCausalLM().to(self.device)
            self.ref_model = DummyCausalLM().to(self.device)
            # Copy weights and freeze ref model
            self.ref_model.load_state_dict(self.policy_model.state_dict())
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        # Fallback to float32 if MPS doesn't support bfloat16 or float16 on some kernels
        if self.device.type == "cpu":
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)

        # Freeze Reference Model (no gradient tracking, evaluation mode)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

    def get_token_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities for the sequence.

        Args:
            model: Policy or Reference model.
            input_ids: Tensor of shape [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            per_token_logps: Tensor of shape [batch_size, seq_len - 1]
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # [B, S-1, V]
        targets = input_ids[:, 1:]           # [B, S-1]

        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return per_token_logps

    @torch.no_grad()
    def get_ref_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log-probabilities using the frozen reference policy."""
        return self.get_token_log_probs(self.ref_model, input_ids, attention_mask)

    def generate_responses(
        self,
        prompt_texts: List[str],
        num_generations: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[int]]:
        """Sample G completions for each prompt.

        Returns:
            all_completions: Flattened list of generated response texts (size = len(prompts) * num_generations)
            all_seq_ids: Tensor of token ids containing prompt + response [B * G, max_len]
            all_attention_mask: Tensor of attention mask [B * G, max_len]
            prompt_lengths: List of token lengths of the prompts for masking
        """
        expanded_prompts = []
        for p in prompt_texts:
            for _ in range(num_generations):
                expanded_prompts.append(p)

        if hasattr(self.tokenizer, "batch_encode"):
            encoded = self.tokenizer.batch_encode(expanded_prompts, padding=True)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
        else:
            encoded = self.tokenizer(
                expanded_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

        prompt_lengths = [int(mask.sum().item()) for mask in attention_mask]

        self.policy_model.eval()
        with torch.no_grad():
            if hasattr(self.policy_model, "generate"):
                gen_out = self.policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0),
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", 0),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", 1),
                )
            else:
                raise ValueError("Model does not implement generate()")

        all_completions = []
        for i, full_ids in enumerate(gen_out):
            p_len = prompt_lengths[i]
            response_ids = full_ids[p_len:]
            text = self.tokenizer.decode(response_ids.tolist())
            all_completions.append(text)

        # Generate attention mask for the generated output
        gen_mask = (gen_out != getattr(self.tokenizer, "pad_token_id", 0)).long()

        return all_completions, gen_out, gen_mask, prompt_lengths
