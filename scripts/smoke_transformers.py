#!/usr/bin/env python3
"""Run RandKV against a real Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from randkv import RandKVConfig
from randkv.transformers import RandKVCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Solve x^2 - 5x + 6 = 0.")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[-1]
    cache = RandKVCache(
        model_config=model.config,
        prompt_length=prompt_length,
        randkv_config=RandKVConfig(
            budget=args.budget,
            buffer_size=args.buffer_size,
            seed=args.seed,
        ),
        request_id="smoke-test",
    )

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = output.shape[-1] - prompt_length
    stats = cache.stats()

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print(
        {
            "device": str(device),
            "generated_tokens": generated,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(generated / elapsed, 3),
            "cache": stats,
        }
    )


if __name__ == "__main__":
    main()
