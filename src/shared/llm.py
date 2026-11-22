"""A small wrapper so every model calls the LLM the same way.

    llm = LLM.from_config(cfg["llm"])
    answers = llm.chat([[{"role": "user", "content": "Hello"}]], temperature=0.0)

Two backends:
  * "vllm"          fast batched generation on Kaggle GPUs (2x T4). Use this.
  * "transformers"  plain Hugging Face generate(); works on CPU too, but slowly.
                    Handy for testing code with a tiny model (Qwen/Qwen3-0.6B).

Qwen3 models "think" before answering by default. We switch that off
(enable_thinking=False) because we want one short answer, fast.
"""

from __future__ import annotations

Messages = list[dict]  # [{"role": "system"|"user"|"assistant", "content": str}, ...]


class LLM:
    def __init__(self, model: str, backend: str = "vllm", dtype: str = "float16",
                 max_new_tokens: int = 128, **backend_kwargs):
        self.model_name = model
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        if backend == "vllm":
            import os

            import torch

            # Kaggle's T4 GPUs are too old for vLLM's default attention there
            # (FlexAttention): it crashes after a few batches with
            # "recompile_limit reached". xFormers attention works on T4, so we
            # pick it for GPUs older than compute capability 8.0 (A100).
            # Set VLLM_ATTENTION_BACKEND yourself to override.
            attention = backend_kwargs.pop("attention_backend", "auto")
            if attention == "auto":
                old_gpu = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8
                attention = "XFORMERS" if old_gpu else None
            if attention and "VLLM_ATTENTION_BACKEND" not in os.environ:
                os.environ["VLLM_ATTENTION_BACKEND"] = attention
                print(f"vLLM attention backend: {attention}")

            from vllm import LLM as VLLM

            self._engine = VLLM(
                model=model,
                dtype=dtype,
                tensor_parallel_size=backend_kwargs.pop("tensor_parallel_size", max(1, torch.cuda.device_count())),
                gpu_memory_utilization=backend_kwargs.pop("gpu_memory_utilization", 0.85),
                max_model_len=backend_kwargs.pop("max_model_len", 4096),
                seed=backend_kwargs.pop("seed", 0),
                **backend_kwargs,
            )
            self._tokenizer = self._engine.get_tokenizer()
        elif backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model)
            self._tokenizer.padding_side = "left"  # needed for batched generation
            torch_dtype = getattr(torch, dtype) if torch.cuda.is_available() else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                model, dtype=torch_dtype, device_map="auto" if torch.cuda.is_available() else None)
            self._model.eval()
            self._batch_size = backend_kwargs.pop("batch_size", 8)
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")

    @classmethod
    def from_config(cls, llm_cfg: dict, **overrides) -> "LLM":
        cfg = {**llm_cfg, **overrides}
        cfg.pop("temperature", None)  # a generation setting, not a model setting
        return cls(**cfg)

    def _to_prompt(self, messages: Messages) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:  # Qwen3: answer directly, no <think> block
            return self._tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def chat(self, conversations: list[Messages], temperature: float = 0.0, n: int = 1,
             seed: int | None = None, max_new_tokens: int | None = None) -> list[list[str]]:
        """Generate answers for many conversations at once.

        Returns one list of n answers per conversation.
        """
        prompts = [self._to_prompt(m) for m in conversations]
        max_new = max_new_tokens or self.max_new_tokens
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(temperature=temperature, n=n, seed=seed, max_tokens=max_new,
                                    top_p=1.0 if temperature == 0 else 0.95)
            outputs = self._engine.generate(prompts, params)
            return [[c.text.strip() for c in out.outputs] for out in outputs]
        return self._hf_generate(prompts, temperature, n, seed, max_new)

    def _hf_generate(self, prompts, temperature, n, seed, max_new):
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        results = []
        for start in range(0, len(prompts), self._batch_size):
            batch = prompts[start:start + self._batch_size]
            enc = self._tokenizer(batch, return_tensors="pt", padding=True).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc, max_new_tokens=max_new, num_return_sequences=n,
                    do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                    top_p=0.95 if temperature > 0 else None, top_k=None,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id)
            texts = self._tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i in range(len(batch)):
                results.append([t.strip() for t in texts[i * n:(i + 1) * n]])
            print(f"  LLM: {min(start + self._batch_size, len(prompts))}/{len(prompts)}", flush=True)
        return results
