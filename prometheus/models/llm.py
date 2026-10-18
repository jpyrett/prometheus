"""Local CausalLM helpers — ported from Lab 1.1 (introduction_to_large_language_models).

Loads AutoTokenizer + AutoModelForCausalLM and exposes chat() / show() for the host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Lab defaults; override with LOCAL_MODEL / HF_MODEL in .env
DEFAULT_MODEL = "Qwen/Qwen3-4B"
# Smaller options (CPU / lighter GPU):
#   Qwen/Qwen3-0.6B
#   Qwen/Qwen3-1.7B


@dataclass
class ChatResult:
    thinking: str
    answer: str


class LocalLLM:
    """Thin wrapper around a Hugging Face CausalLM + chat template."""

    def __init__(self, model_name: str | None = None, seed: int = 42):
        self.model_name = (
            model_name
            or os.getenv("LOCAL_MODEL")
            or os.getenv("HF_MODEL")
            or DEFAULT_MODEL
        )
        print(f"Loading {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map="auto",
        )
        set_seed(seed)
        print("Loaded!")

    def chat(
        self,
        prompt: str,
        *,
        enable_thinking: bool = False,
        max_new_tokens: int = 256,
        **gen_kwargs,
    ) -> ChatResult:
        """Send one user turn through the model's chat template.

        Returns ChatResult(thinking, answer). thinking is set only when
        enable_thinking=True and the model emits a </think> block.
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        sampling = (
            dict(temperature=0.6, top_p=0.95, top_k=20)
            if enable_thinking
            else dict(temperature=0.7, top_p=0.8, top_k=20)
        )
        sampling.update(gen_kwargs)

        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **sampling,
        )
        output_ids = generated[0][model_inputs.input_ids.shape[1] :].tolist()

        think_close = self.tokenizer.convert_tokens_to_ids("</think>")
        try:
            split = len(output_ids) - output_ids[::-1].index(think_close)
        except ValueError:
            split = 0

        thinking = self.tokenizer.decode(
            output_ids[:split], skip_special_tokens=True
        ).strip()
        answer = self.tokenizer.decode(
            output_ids[split:], skip_special_tokens=True
        ).strip()
        return ChatResult(thinking=thinking, answer=answer)

    @staticmethod
    def show(result: ChatResult) -> None:
        """Pretty-print a ChatResult."""
        if result.thinking:
            print(f"{YELLOW}Thinking:{RESET} {result.thinking}")
        print(f"{GREEN}Answer:{RESET} {result.answer}")
