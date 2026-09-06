#!/usr/bin/env python3
"""Compare dense and RandKV Transformers generation on one local device."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from randkv import RandKVConfig
from randkv.transformers import RandKVCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Solve x^2 - 5x + 6 = 0.")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hardware-label")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.max_new_tokens < 1 or args.warmup_tokens < 1:
        parser.error("token counts must be positive")
    return args


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_once(
    *,
    model: Any,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    max_new_tokens: int,
    cache: RandKVCache | None,
) -> tuple[dict[str, float | int], torch.Tensor]:
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
        )
    synchronize(device)
    elapsed = time.perf_counter() - started
    generated = output.shape[-1] - inputs["input_ids"].shape[-1]
    return (
        {
            "generated_tokens": generated,
            "elapsed_seconds": round(elapsed, 6),
            "tokens_per_second": round(generated / elapsed, 6),
        },
        output,
    )


def summarize(runs: list[dict[str, float | int]]) -> dict[str, float]:
    throughputs = [float(run["tokens_per_second"]) for run in runs]
    return {
        "median_tokens_per_second": round(statistics.median(throughputs), 6),
        "min_tokens_per_second": round(min(throughputs), 6),
        "max_tokens_per_second": round(max(throughputs), 6),
    }


def main() -> None:
    args = parse_args()
    device = choose_device()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    inputs = dict(tokenizer(args.prompt, return_tensors="pt").to(device))
    prompt_length = inputs["input_ids"].shape[-1]
    config = RandKVConfig(
        budget=args.budget,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )

    for mode in ("dense", "randkv"):
        cache = (
            RandKVCache(
                model_config=model.config,
                prompt_length=prompt_length,
                randkv_config=config,
                request_id=f"warmup-{mode}",
            )
            if mode == "randkv"
            else None
        )
        run_once(
            model=model,
            inputs=inputs,
            device=device,
            max_new_tokens=args.warmup_tokens,
            cache=cache,
        )

    runs: dict[str, list[dict[str, float | int]]] = {
        "dense": [],
        "randkv": [],
    }
    last_cache: RandKVCache | None = None
    outputs: dict[str, torch.Tensor] = {}
    for trial in range(args.trials):
        order = ("dense", "randkv") if trial % 2 == 0 else ("randkv", "dense")
        for mode in order:
            cache = (
                RandKVCache(
                    model_config=model.config,
                    prompt_length=prompt_length,
                    randkv_config=config,
                    request_id=f"trial-{trial}",
                )
                if mode == "randkv"
                else None
            )
            run, output = run_once(
                model=model,
                inputs=inputs,
                device=device,
                max_new_tokens=args.max_new_tokens,
                cache=cache,
            )
            run["trial"] = trial
            runs[mode].append(run)
            outputs[mode] = output
            if cache is not None:
                last_cache = cache

    dense_summary = summarize(runs["dense"])
    randkv_summary = summarize(runs["randkv"])
    ratio = (
        randkv_summary["median_tokens_per_second"]
        / dense_summary["median_tokens_per_second"]
    )
    result = {
        "schema_version": 1,
        "kind": "single_request_transformers_microbenchmark",
        "claim_scope": (
            "Local compatibility and adapter-overhead evidence only; not a "
            "vLLM serving or model-quality benchmark."
        ),
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": args.hardware_label,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "prompt_tokens": prompt_length,
        "generated_tokens": args.max_new_tokens,
        "warmup_tokens": args.warmup_tokens,
        "trials": args.trials,
        "budget": args.budget,
        "buffer_size": args.buffer_size,
        "seed": args.seed,
        "runs": runs,
        "summary": {
            "dense": dense_summary,
            "randkv": randkv_summary,
            "randkv_over_dense_throughput": round(ratio, 6),
            "last_outputs_identical": bool(
                torch.equal(outputs["dense"], outputs["randkv"])
            ),
        },
        "last_randkv_cache": asdict(last_cache.stats()) if last_cache else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
