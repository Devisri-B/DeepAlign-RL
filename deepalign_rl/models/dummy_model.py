"""Lightweight self-contained causal LM and tokenizer for testing, CI, and dry runs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DummyTokenizer:
    """Character/subword-level lightweight tokenizer for testing."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [ord(c) % (self.vocab_size - 4) + 4 for c in text]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        chars = []
        for tid in token_ids:
            if tid < 4:
                continue
            char_code = (tid - 4)
            # Map back to ASCII printable where possible
            if 32 <= char_code <= 126:
                chars.append(chr(char_code))
            else:
                chars.append("?")
        return "".join(chars)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> dict:
        encoded_list = [self.encode(t) for t in texts]
        max_len = max(len(ids) for ids in encoded_list) if encoded_list else 0
        if max_length is not None:
            max_len = min(max_len, max_length)

        input_ids = []
        attention_mask = []

        for ids in encoded_list:
            if len(ids) > max_len:
                truncated = ids[:max_len]
            else:
                truncated = ids

            pad_len = max_len - len(truncated)
            ids_padded = truncated + [self.pad_token_id] * pad_len
            mask = [1] * len(truncated) + [0] * pad_len

            input_ids.append(ids_padded)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DummyCausalLM(nn.Module):
    """Small genuine PyTorch Transformer LM for testing loss & rollouts."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        seq_len = input_ids.size(1)
        # Causal mask
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1)

        embeds = self.embedding(input_ids)
        # Pass through transformer with causal mask
        hidden = self.transformer(embeds, mask=causal_mask)
        logits = self.lm_head(hidden)

        # Output struct matching HF CausalLMOutput
        class Output:
            pass

        out = Output()
        out.logits = logits
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation for testing."""
        curr_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(curr_ids).logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
        return curr_ids
