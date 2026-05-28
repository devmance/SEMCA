"""
Model loader with activation extraction hooks.

Provides a uniform interface for running open-weight HuggingFace models
on Apple Silicon (MPS), CUDA, or CPU, with batched activation/attention/logit
extraction for downstream theory-internals measurements.

The loader returns an `InternalsRecord` per forward pass containing:
  - hidden_states: tuple of (n_layers + 1) tensors, each (batch, seq, hidden)
  - attentions: tuple of n_layers tensors, each (batch, n_heads, seq, seq)
  - logits: (batch, seq, vocab) final-layer logits
  - input_ids, attention_mask: for reference

Designed for fast iteration on small models (1B-9B). For larger models,
swap in nnsight or transformer-lens for memory-efficient hook management.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    """Pick the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class InternalsRecord:
    """One forward-pass result with all activations exposed."""
    input_ids: torch.Tensor       # (batch, seq)
    attention_mask: torch.Tensor  # (batch, seq)
    hidden_states: tuple          # (n_layers + 1) × (batch, seq, hidden)
    attentions: tuple             # n_layers × (batch, n_heads, seq, seq)
    logits: torch.Tensor          # (batch, seq, vocab)
    prompt: str
    response: str
    model_name: str

    @property
    def n_layers(self) -> int:
        return len(self.attentions)

    @property
    def n_heads(self) -> int:
        return self.attentions[0].shape[1]

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.hidden_states[0].shape[-1]


class ModelInternalsRunner:
    """
    Load a HuggingFace causal LM and run it with full internals extraction.

    Usage:
        runner = ModelInternalsRunner("microsoft/Phi-3-mini-4k-instruct")
        record = runner.run_with_internals("What is consciousness?")
        # record.hidden_states[3]  → 4th-layer hidden states
        # record.attentions[3]     → 4th-layer attention patterns
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 256,
        device_map: Optional[str] = None,
    ):
        """
        Load a causal LM with full internals extraction.

        Args:
            device: 'cuda' | 'mps' | 'cpu' — used only for single-GPU loads.
            device_map: HuggingFace device_map override. For multi-GPU
                (e.g. 2× H100 hosting Llama 70B fp16), pass 'auto' to
                distribute layers across all visible GPUs. If None,
                defaults to the picked device for single-GPU cases or
                'auto' when multiple CUDA devices are visible.
        """
        self.model_name = model_name
        self.device = device or pick_device()
        # MPS prefers fp16 / bfloat16; CPU defaults to fp32; CUDA fp16
        if dtype is None:
            dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        # Auto-pick device_map: multi-GPU → 'auto' (HF sharding); single → device name
        if device_map is None:
            if self.device == "cuda" and torch.cuda.device_count() > 1:
                device_map = "auto"
            else:
                device_map = self.device

        print(f"Loading {model_name} on {self.device} ({dtype}, device_map={device_map})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            output_hidden_states=True,
            output_attentions=True,
        )
        self.model.eval()
        # Note: for sharded models, parameter inventory crosses devices but
        # the count is still correct
        print(f"  Model loaded: {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B params")

    @torch.no_grad()
    def run_with_internals(self, prompt: str, max_new_tokens: Optional[int] = None) -> InternalsRecord:
        """
        Generate a response to `prompt`, then re-run the full (prompt + response)
        sequence with hidden_states + attentions enabled. Returns InternalsRecord.

        We generate first, then re-encode, because generation with output_attentions
        materializes O(n_layers × n_heads × seq²) tensors per step which OOMs quickly.
        Re-encoding the final sequence once is much cheaper.
        """
        n = max_new_tokens or self.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=n,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response_ids = gen_out[0, inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Re-encode full (prompt + response) for clean internals extraction
        full_text = prompt + response_text
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        out = self.model(
            **full_inputs,
            output_hidden_states=True,
            output_attentions=True,
        )

        return InternalsRecord(
            input_ids=full_inputs.input_ids,
            attention_mask=full_inputs.attention_mask,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
            logits=out.logits,
            prompt=prompt,
            response=response_text,
            model_name=self.model_name,
        )

    @torch.no_grad()
    def encode_with_internals(self, text: str) -> InternalsRecord:
        """Run model on existing text without generating new tokens.
        Use this to extract attention activations and hidden states from
        text you already have, for substrate-agnostic theory scoring."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )
        return InternalsRecord(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
            logits=out.logits,
            prompt="",
            response=text,
            model_name=self.model_name,
        )
